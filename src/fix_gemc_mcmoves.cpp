#include "fix_gemc.h"

#include "atom.h"
#include "atom_vec.h"
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

// this must be lower than MAXENERGYSIGNAL
// by a large amount, so that it is still
// less than total energy when negative
// energy contributions are added to MAXENERGYSIGNAL

static constexpr double MAXENERGYTEST = 1.0e50;

/* ----------------------------------------------------------------------
  Shrink/expand boxes (always requires full energy)
------------------------------------------------------------------------- */
void FixGEMC::attempt_volume_change()
{
  // current volume
  double Lx = domain->boxhi[0]-domain->boxlo[0];
  double Ly = domain->boxhi[1]-domain->boxlo[1];
  double Lz = domain->boxhi[2]-domain->boxlo[2];

  double volume = Lx*Ly*Lz;

  // sample volume change from world 0 comm 0
  double dvolume;
  if (mycomm == 0) {
    double min_volume;  
    MPI_Allreduce(&min_volume, &volume, 1, MPI_DOUBLE, MPI_MIN, comm_replica);
    // have one world sample volume change
    if (myworld == 0) {
      dvolume = 2.0*min_volume;
      // make sure volume change less than available volume
      while (fabs(dvolume) >= min_volume)
        dvolume = (2.0*random->uniform()-1.0)*max_volume;
    }
    /// broadcast volume change to other world
    MPI_Bcast(&dvolume, 1, MPI_DOUBLE, 0, comm_replica);
    // have one world invert the volume change so total volume conserved
    if (myworld) dvolume *= -1.0;
  }

  // broadcast volume change to all worlds
  MPI_Bcast(&dvolume, 1, MPI_DOUBLE, 0, world);

  // attempt to change volume
  double scale_length = pow((volume-dvolume)/volume, 1/domain->dimension);

  // find center point of box
  double xm = (domain->boxhi[0]+domain->boxlo[0])*0.5;
  double ym = (domain->boxhi[1]+domain->boxlo[1])*0.5;
  double zm = (domain->boxhi[2]+domain->boxlo[2])*0.5;

  // change in potential due to volume change
  double dU_volume = atom->nlocal*force->boltz*box_temp*
              log((volume+dvolume)/volume);
  // current system energy
  double energy_before = energy_stored;
  // scale the particle positions (will revert if not accepted)
  scale_positions(xm,ym,zm,scale_length);
  // (possible) future system energy
  double energy_after = energy_full();

  // get total energy from each partition
  double dU;
  if (mycomm == 0) {
    double idU = energy_after-energy_before-dU_volume;
    // sum change in full energy across each box
    MPI_Allreduce(&dU, &idU, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
  }
  // bcast decision to rest of world
  MPI_Bcast(&dU, 1, MPI_DOUBLE, 0, world);

  // evaluate probability
  double prob = MIN(exp(-beta*dU),1.0);
  // scale back particle positions if volume change rejected
  // random_volume should give same number across all procs across boxes
  if (prob < random_vol->uniform()) 
    scale_positions(xm,ym,zm,1.0/scale_length);
  else
  {
    // store new energy
    energy_stored = energy_after;

    // shrink/expand box lengths wrt to center
    domain->boxhi[0] = xm + Lx*0.5*scale_length;
    domain->boxhi[1] = ym + Ly*0.5*scale_length;
    domain->boxhi[2] = zm + Lz*0.5*scale_length;

    domain->boxlo[0] = xm - Lx*0.5*scale_length;
    domain->boxlo[1] = ym - Ly*0.5*scale_length;
    domain->boxlo[2] = zm - Lz*0.5*scale_length;

    domain->boxhi[0] = xm + Lx*0.5*scale_length;
    domain->boxhi[1] = ym + Ly*0.5*scale_length;
    domain->boxhi[2] = zm + Lz*0.5*scale_length;

    domain->boxlo[0] = xm - Lx*0.5*scale_length;
    domain->boxlo[1] = ym - Ly*0.5*scale_length;
    domain->boxlo[2] = zm - Lz*0.5*scale_length;

    // reset box and subbox dimensions
    domain->set_global_box();
    domain->set_local_box();

    // reacquire domain bounds
    xlo = domain->boxlo[0];
    xhi = domain->boxhi[0];
    ylo = domain->boxlo[1];
    yhi = domain->boxhi[1];
    zlo = domain->boxlo[2];
    zhi = domain->boxhi[2];

    // reacquire subdomain bounds
    if (triclinic_flag) {
      sublo = domain->sublo_lamda;
      subhi = domain->subhi_lamda;
    } else {
      sublo = domain->sublo;
      subhi = domain->subhi;
    }

  }
}

/* ----------------------------------------------------------------------
  Attempt atom exchange
------------------------------------------------------------------------- */
void FixGEMC::attempt_atomic_exchange_full()
{
  // choose which box sends particle and which box receives
  int sender;
  if (mycomm == 0) {
    double drand = random->uniform();
    double dmean;
    MPI_Allreduce(&drand, &dmean, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
    dmean *= 0.5;
    int iparticle = -1;
    if (drand > dmean) sender = 1;
    else sender = 0;
  }
  MPI_Bcast(&sender, 1, MPI_INT, 0, world);

  // get change in potential energy
  double dU = 0.0;
  double energy_after;
  double energy_before = energy_stored;

  // first choose which atom to send and delete
  // temporary store
  double q_iatom;
  int mask_iatom;

  int iatom;
  int iatom_type; // record atom type
  double v_iatom[3];
  if (sender) {
    // pick one atom randomly from all atoms in system
    // only one proc will actually delete atom
    update_gas_atoms_list();
    iatom = pick_random_gas_atom();
    double idU = 0.0;

    if (iatom >= 0) {
      mask_iatom = atom->mask[iatom]; // store particle mask
      iatom_type = atom->type[iatom]; // for insertion
      atom->mask[iatom] = exclusion_group_bit;
      // check if charged
      if (iatom >= 0 && q_flag) {
        q_iatom = atom->q[iatom];
        atom->q[iatom] = 0.0;
      }

      // store velocity
      v_iatom[0] = atom->v[iatom][0];
      v_iatom[1] = atom->v[iatom][1];
      v_iatom[2] = atom->v[iatom][2];

      if (force->kspace) force->kspace->qsum_qsq();
      if (force->pair->tail_flag) force->pair->reinit();
      energy_after = energy_full();
      idU = energy_after-energy_before;
    }
    MPI_Allreduce(&idU, &dU, 1, MPI_DOUBLE, MPI_SUM, world);
  }

  // tell other box the atome type and velocity it's receiving
  if (mycomm == 0 && sender) {
    MPI_Bcast(&iatom_type, 1, MPI_INT, myworld, comm_replica);
    MPI_Bcast(&v_iatom, 3, MPI_DOUBLE, myworld, comm_replica);
  }
  MPI_Bcast(&iatom_type, 1, MPI_INT, 0, world);
  MPI_Bcast(&v_iatom, 3, MPI_DOUBLE, 0, world);

  // then send the atom to other box
  int proc_flag = 0;
  if (!sender) {
    double lamda[3], coord[3];
    if (mycomm == 0) {
      if (triclinic_flag) {
        lamda[0] = random->uniform();
        lamda[1] = random->uniform();
        lamda[2] = random->uniform();

        // wasteful, but necessary

        if (lamda[0] == 1.0) lamda[0] = 0.0;
        if (lamda[1] == 1.0) lamda[1] = 0.0;
        if (lamda[2] == 1.0) lamda[2] = 0.0;

        domain->lamda2x(lamda,coord);
      } else {
        coord[0] = xlo + random->uniform() * (xhi-xlo);
        coord[1] = ylo + random->uniform() * (yhi-ylo);
        coord[2] = zlo + random->uniform() * (zhi-zlo);
      }
    } // END mycomm

    MPI_Bcast(&coord, 3, MPI_DOUBLE, 0, world);

    // find proc with this coordinate
    if (triclinic_flag) {
      if (lamda[0] >= sublo[0] && lamda[0] < subhi[0] &&
          lamda[1] >= sublo[1] && lamda[1] < subhi[1] &&
          lamda[2] >= sublo[2] && lamda[2] < subhi[2]) proc_flag = 1;
    } else {
      domain->remap(coord);
      if (!domain->inside(coord))
        error->one(FLERR,"Fix gemc put atom outside box");
      if (coord[0] >= sublo[0] && coord[0] < subhi[0] &&
          coord[1] >= sublo[1] && coord[1] < subhi[1] &&
          coord[2] >= sublo[2] && coord[2] < subhi[2]) proc_flag = 1;
    } // END if triclinic

    if (proc_flag) {
      atom->avec->create_atom(iatom_type,coord);
      int jatom = atom->nlocal - 1;

      // add to groups
      // optionally add to type-based groups

      atom->mask[jatom] = mask_iatom;
      atom->v[jatom][0] = v_iatom[0];
      atom->v[jatom][1] = v_iatom[1];
      atom->v[jatom][2] = v_iatom[2];
      if (q_flag) atom->q[jatom] = q_iatom;
      modify->create_attribute(jatom);
    } // END if proc_flag

    atom->natoms++;

    // TODO: What is happening here
    //if (atom->tag_enable) {
    //  atom->tag_extend();
    //  if (atom->map_style != Atom::MAP_NONE) atom->map_init();
    //}

    atom->nghost = 0;
    if (triclinic_flag) domain->x2lamda(atom->nlocal);
    comm->borders();
    if (triclinic_flag) domain->lamda2x(atom->nlocal+atom->nghost);
    if (force->kspace) force->kspace->qsum_qsq();
    if (force->pair->tail_flag) force->pair->reinit();
    energy_after = energy_full();
    dU = energy_after-energy_before;
  } // END if sender

  // evalute probability for exchange
  double dU_all;
  int success;
  if (comm->me == 0) {
    MPI_Allreduce(&dU,&dU_all,1,MPI_DOUBLE,MPI_SUM,comm_replica);
    double volume = (xhi-xlo)*(yhi-ylo)*(zhi-zlo);
    double NV;
    if (sender) NV = volume/atom->natoms;
    else NV = (atom->natoms)/volume;
    double allNV;
    MPI_Allreduce(&NV,&allNV,1,MPI_DOUBLE,MPI_PROD,comm_replica);

    dU_all += (box_temp*force->boltz*log(allNV));
    double prob = MIN(exp(-beta*dU_all),1.0);
    if (prob >= random->uniform()) success = 1;
    MPI_Bcast(&success, 1, MPI_INT, 0, comm_replica);
  }
  MPI_Bcast(&success, 1, MPI_INT, 0, world);
  
  if (sender) {
    // delete iatom
    if (success) {
      if (iatom >= 0) {
        atom->avec->copy(atom->nlocal-1,iatom,1);
        atom->nlocal--;
      }
      atom->natoms--;
      if (atom->map_style != Atom::MAP_NONE) atom->map_init();
      energy_stored = energy_after;
    // revert iatom (do not delete)
    } else {
      if (iatom >= 0) {
        atom->mask[iatom] = mask_iatom;
        if (q_flag) atom->q[iatom] = q_iatom;
      }
      if (force->kspace) force->kspace->qsum_qsq();
      if (force->pair->tail_flag) force->pair->reinit();
      energy_stored = energy_before;
    }
  } else {
    // accept newly inserted iatomthere
    if (success) {
      energy_stored = energy_after;
    // remove newly inserted iatom
    } else {
      atom->natoms--;
      if (proc_flag) atom->nlocal--;
      if (force->kspace) force->kspace->qsum_qsq();
      if (force->pair->tail_flag) force->pair->reinit();
      energy_stored = energy_before;
    }
  }

  // update counts
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGEMC::attempt_atomic_translation_full()
{
  if (natom_local == 0) return;

  double energy_before = energy_stored;

  int i = pick_random_gas_atom();

  double **x = atom->x;
  double xtmp[3];

  xtmp[0] = xtmp[1] = xtmp[2] = 0.0;

  tagint tmptag = -1;

  if (i >= 0) {

    double rsq = 1.1;
    double rx,ry,rz;
    rx = ry = rz = 0.0;
    double coord[3];
    while (rsq > 1.0) {
      rx = 2*random->uniform() - 1.0;
      ry = 2*random->uniform() - 1.0;
      rz = 2*random->uniform() - 1.0;
      rsq = rx*rx + ry*ry + rz*rz;
    }
    coord[0] = x[i][0] + displace*rx;
    coord[1] = x[i][1] + displace*ry;
    coord[2] = x[i][2] + displace*rz;

    if (!domain->inside_nonperiodic(coord))
      error->one(FLERR,"Fix gemc put atom outside box");
    xtmp[0] = x[i][0];
    xtmp[1] = x[i][1];
    xtmp[2] = x[i][2];
    x[i][0] = coord[0];
    x[i][1] = coord[1];
    x[i][2] = coord[2];

    tmptag = atom->tag[i];
  }

  double energy_after = energy_full();

  if (energy_after < MAXENERGYTEST &&
      random->uniform() <
      exp(beta*(energy_before - energy_after))) {
    energy_stored = energy_after;
  } else {

    tagint tmptag_all;
    MPI_Allreduce(&tmptag,&tmptag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);

    double xtmp_all[3];
    MPI_Allreduce(&xtmp,&xtmp_all,3,MPI_DOUBLE,MPI_SUM,world);

    for (int i = 0; i < natom_local; i++) {
      if (tmptag_all == atom->tag[i]) {
        x[i][0] = xtmp_all[0];
        x[i][1] = xtmp_all[1];
        x[i][2] = xtmp_all[2];
      }
    }
    energy_stored = energy_before;
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixGEMC::attempt_molecule_translation_full()
{
  if (natom_local == 0) return;

  tagint translation_molecule = pick_random_gas_molecule();
  if (translation_molecule == -1) return;

  double energy_before = energy_stored;

  double **x = atom->x;
  double rx,ry,rz;
  double com_displace[3],coord[3];
  double rsq = 1.1;
  while (rsq > 1.0) {
    rx = 2*random->uniform() - 1.0;
    ry = 2*random->uniform() - 1.0;
    rz = 2*random->uniform() - 1.0;
    rsq = rx*rx + ry*ry + rz*rz;
  }
  com_displace[0] = displace*rx;
  com_displace[1] = displace*ry;
  com_displace[2] = displace*rz;

  for (int i = 0; i < natom_local; i++) {
    if (atom->molecule[i] == translation_molecule) {
      x[i][0] += com_displace[0];
      x[i][1] += com_displace[1];
      x[i][2] += com_displace[2];
      if (!domain->inside_nonperiodic(x[i]))
        error->one(FLERR,"Fix gemc put atom outside box");
    }
  }

  double energy_after = energy_full();

  if (energy_after < MAXENERGYTEST &&
      random->uniform() <
      exp(beta*(energy_before - energy_after))) {
    energy_stored = energy_after;
  } else {
    energy_stored = energy_before;
    for (int i = 0; i < natom_local; i++) {
      if (atom->molecule[i] == translation_molecule) {
        x[i][0] -= com_displace[0];
        x[i][1] -= com_displace[1];
        x[i][2] -= com_displace[2];
      }
    }
  }
  update_gas_atoms_list();
}

