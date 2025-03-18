// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier, Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include "fix_gemc.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"
#include "universe.h"
#include "update.h"
#include "variable.h"

// for molecule
#include "angle.h"
#include "bond.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

// large energy value used to signal overlap

static constexpr double MAXENERGYSIGNAL = 1.0e100;

/* ---------------------------------------------------------------------- */

// not sure what all these nullptrs are for
FixGEMC::FixGEMC(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg), commbuf(nullptr)
{

  if (narg < 12) utils::missing_cmd_args(FLERR, "fix gcmc", error);

  // must have only two boxes

  if (universe->nworlds != 2) error->universe_all(FLERR, "Must use exactly two partitions");

  // required args

  nevery = utils::inumeric(FLERR,     arg[3], false, lmp);
  ntranslate = utils::inumeric(FLERR, arg[4], false, lmp);
  nrotate = utils::inumeric(FLERR,    arg[5], false, lmp);
  nexchange = utils::inumeric(FLERR,  arg[6], false, lmp);
  nvolume = utils::inumeric(FLERR,    arg[7], false, lmp);
  box_temp = utils::numeric(FLERR,    arg[8], false, lmp);
  displace = utils::numeric(FLERR,    arg[9], false, lmp);
  max_volume = utils::numeric(FLERR,  arg[10], false, lmp);
  seed = utils::inumeric(FLERR,       arg[11], false, lmp);

  // comm_replica = communicator between proc 0s across boxes

  int color = comm->me;
  MPI_Comm_split(universe->uworld, color, 0, &comm_replica);

  // same RNG for each replica for volume MC moves
  random = new RanPark(lmp,seed+universe->iworld); // general purpose rng
  random_mc = new RanPark(lmp,seed+2); // sync which type of move to make
  random_vol = new RanPark(lmp,seed+3); // sync volume changes

  // read options from end of input line

  options(narg-12,&arg[12]);
}

/* ---------------------------------------------------------------------- */

FixGEMC::~FixGEMC()
{
  MPI_Comm_free(&comm_replica);
  memory->destroy(commbuf);
}

/* ---------------------------------------------------------------------- */

int FixGEMC::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGEMC::init()
{

  // for comm
  myworld = universe->iworld;
  mycomm = comm->me;
  nprocs = comm->nprocs;

  // determine probability of each step during pre_exchange

  // set probabilities for MC moves
  if (!molecule_flag) nrotate = 0;

  // total moves is a double to avoid type casting later
  nmoves = nvolume + nexchange + ntranslate + nrotate;

  double p_exchange  = nexchange/nmoves;
  double p_volume    = nvolume/nmoves;
  double p_translate = ntranslate/nmoves;
  double p_rotate    = nrotate/nmoves;

  // normalize probabilities
  double p_total = p_exchange + p_volume + p_translate + p_rotate;
  
  // compute cummulative probabilites
  pc_exchange = p_exchange/p_total;
  pc_volume = (p_volume+p_exchange)/p_total;
  if (molecule_flag) {
    pc_translate = (p_volume+p_exchange+p_translate)/p_total;
    pc_rotate = 1.0;
  } else {
    pc_translate = 1.0;
    pc_rotate = 0.0;
  }

  // set to full energy?

  if (!full_flag) {
    if ((force->kspace) ||
        (force->pair == nullptr) ||
        (force->pair->single_enable == 0) ||
        (force->pair_match("^hybrid",0)) ||
        (force->pair_match("^eam",0)) ||
        (force->pair->tail_flag)) {
      full_flag = true;
      if (mycomm == 0)
        error->warning(FLERR,"Fix gemc using full_energy option");
    }
  }

  if (full_flag) c_pe = modify->compute[modify->find_compute("thermo_pe")];

  // check if atoms charged
  q_flag = atom->q_flag;

  // pre compute

  beta = 1.0/(force->boltz*box_temp);

  // get domain dim

  triclinic_flag = domain->triclinic;

  xlo = domain->boxlo[0];
  xhi = domain->boxhi[0];
  ylo = domain->boxlo[1];
  yhi = domain->boxhi[1];
  zlo = domain->boxlo[2];
  zhi = domain->boxhi[2];

  // get subdomain
  if (triclinic_flag) {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  } else {
    sublo = domain->sublo;
    subhi = domain->subhi;
  }

  // create unique group name for atoms to be excluded for particle exchange
  // keeps temporarily deleted particles from being added in potential energy calc

  // id from fix
  auto group_id = std::string("FixGEMC:gemc_exclusion_group:") + id;
  group->assign(group_id + " subtract all all");
  exclusion_group = group->find(group_id);
  if (exclusion_group == -1)
    error->all(FLERR,"Could not find fix gemc exclusion group ID");
  exclusion_group_bit = group->bitmask[exclusion_group];

  // neighbor list exclusion setup
  // turn off interactions between group all and the exclusion group

  neighbor->modify_params(fmt::format("exclude group {} all",group_id));
}

/* ----------------------------------------------------------------------
   attempt Monte Carlo translations, rotations, insertions, and deletions
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct

   gcmc + extra volume step and modified insertion/deletion
------------------------------------------------------------------------- */

void FixGEMC::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  // three steps in GEMC:
  // 1) translate particles within each box
  // 2) exchange particles between boxes
  // 3) change box volume + scale particle positions

  // cannot randomly choose which MC move to make since exchanges and volume
  // ... changes are coupled
  // do translations/rotations first
  // no communication needed between boxes

  if (full_flag) {
    energy_stored = energy_full();
    //if (overlap_flag && energy_stored > MAXENERGYTEST)
    //    error->warning(FLERR,"Energy of old configuration in "
    //                   "fix gcmc is > MAXENERGYTEST.");

    for (int i = 0; i < nmoves; i++) {
      double imove = random_mc->uniform();
      if (molecule_flag) { // TODO : add molecule counterpart
        if (imove < pc_exchange) ;//attempt_molecule_exchange_full();
        else if (imove < pc_volume) attempt_volume_change();
        else if (imove < pc_translate) ;//attempt_molecule_translation_full();
        else ;//attempt_molecule_rotation_full();
      } else {
        if (imove < pc_exchange) attempt_atomic_exchange_full();
        else if (imove < pc_volume) attempt_volume_change();
        else attempt_atomic_translation_full();
      }
    }
  } // TODO: Add not full option
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixGEMC::options(int narg, char **arg)
{
  // initialize flags
  molecule_flag = 0;
  full_flag = 1;

  int iarg = 0;
  //while (iarg < narg) {
  //}

}

/* ----------------------------------------------------------------------
  Scale new particle positions according to volume change
------------------------------------------------------------------------- */
void FixGEMC::scale_positions(const double xm, const double ym, const double zm, const double scale)
{
  double **x = atom->x;

  for (int i = 0; i < natom_total; i++) {
    x[i][0] = (x[i][0]-xm)/scale+xm;
    x[i][1] = (x[i][1]-ym)/scale+ym;
    x[i][2] = (x[i][2]-zm)/scale+zm;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixGEMC::pick_random_gas_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (natom_total*random->uniform());
  if ((iwhichglobal >= natom_lower) &&
      (iwhichglobal < natom_lower + natom_local)) {
    i = iwhichglobal - natom_lower;
  }

  return i;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

tagint FixGEMC::pick_random_gas_molecule()
{
  int iwhichglobal = static_cast<int> (natom_local*random->uniform());
  tagint gas_molecule_id = 0;
  if ((iwhichglobal >= natom_lower) &&
      (iwhichglobal < natom_lower + natom_local)) {
    gas_molecule_id = iwhichglobal - natom_lower;
  }

  tagint gas_molecule_id_all = 0;
  MPI_Allreduce(&gas_molecule_id,&gas_molecule_id_all,1,
                MPI_LMP_TAGINT,MPI_MAX,world);

  return gas_molecule_id_all;
}

/* ----------------------------------------------------------------------
   update per-proc atom count
   assume all atoms are candidates for MC moves
------------------------------------------------------------------------- */

void FixGEMC::update_gas_atoms_list()
{
  natom_local = atom->nlocal;
  natom_total = atom->nlocal + atom->nghost;

  // ngas is total atoms in whole system
  MPI_Allreduce(&natom_local,&natom_total,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&natom_local,&natom_lower,1,MPI_INT,MPI_SUM,world);
  natom_lower -= natom_local;
}

/* ----------------------------------------------------------------------
   compute system potential energy
------------------------------------------------------------------------- */

double FixGEMC::energy_full()
{
  int imolecule;

  if (triclinic_flag) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0; // ??
  comm->borders();
  if (triclinic_flag) domain->lamda2x(atom->nlocal+atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);
  int eflag = 1;
  int vflag = 0;

  // if overlap check requested, if overlap,
  // return signal value for energy

  // TODO : TEMPORARY
  int overlap_flag = 1; // manually call for now
  double overlap_cutoffsq = 0.0;

  if (overlap_flag) {
    int overlaptestall;
    int overlaptest = 0;
    double delx,dely,delz,rsq;
    double **x = atom->x;
    tagint *molecule = atom->molecule;
    int nall = atom->nlocal + atom->nghost;
    for (int i = 0; i < atom->nlocal; i++) {
      if (molecule_flag)
        imolecule = molecule[i];
      for (int j = i+1; j < nall; j++) {
        if (molecule_flag)
          if (imolecule == molecule[j]) continue;

        delx = x[i][0] - x[j][0];
        dely = x[i][1] - x[j][1];
        delz = x[i][2] - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < overlap_cutoffsq) {
          overlaptest = 1;
          break;
        }
      }
      if (overlaptest) break;
    }
    MPI_Allreduce(&overlaptest, &overlaptestall, 1, MPI_INT, MPI_MAX, world);
    if (overlaptestall) return MAXENERGYSIGNAL;
  }

  // clear forces so they don't accumulate over multiple
  // calls within fix gcmc timestep, e.g. for fix shake

  size_t nbytes = sizeof(double) * (atom->nlocal + atom->nghost);
  if (nbytes) memset(&atom->f[0][0],0,3*nbytes);

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) force->kspace->compute(eflag,vflag);

  if (modify->n_post_force_any) modify->post_force(vflag);

  // NOTE: all fixes with energy_global_flag set and which
  //   operate at pre_force() or post_force()
  //   and which user has enabled via fix_modify energy yes,
  //   will contribute to total MC energy via pe->compute_scalar()

  update->eflag_global = update->ntimestep;
  double total_energy = c_pe->compute_scalar();

  return total_energy;
}






