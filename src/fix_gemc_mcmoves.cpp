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
#include "irregular.h"
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
void FixGEMC::attempt_volume_change_full()
{
  nvolume_attempts++;
  // current volume
  double Lx = xhi-xlo;
  double Ly = yhi-ylo;
  double Lz = zhi-zlo;
  double volume = Lx*Ly*Lz;

  // sample volume change from world 0 comm 0
  double dvolume;
  if (mycomm == 0) {
    double min_volume;  
    MPI_Allreduce(&volume, &min_volume, 1, MPI_DOUBLE, MPI_MIN, comm_replica);
    // have one world sample volume change
    if (myworld == 0) {
      dvolume = 2.0*min_volume;
      // make sure volume change less than available volume
      while (fabs(dvolume) >= min_volume)
        dvolume = (2.0*random->uniform()-1.0)*max_volume;
    }
    // TODO : TEMPORARY
    //dvolume = max_volume;

    /// broadcast volume change to other world
    MPI_Bcast(&dvolume, 1, MPI_DOUBLE, 0, comm_replica);
    // have one world invert the volume change so total volume conserved
    if (myworld) dvolume *= -1.0;
  }

  // broadcast volume change to all worlds
  MPI_Bcast(&dvolume, 1, MPI_DOUBLE, 0, world);

  // DEBUG : Check that each comm in each partition has corect volume + volume change
  //printf("xhi: %g, %g, %g\n", xhi, yhi, zhi);
  //printf("%i/%i - dvolume: %g / %g\n", myworld, mycomm, dvolume, volume);
  //error->one(FLERR,"ck");

  // attempt to change volume
  double fvolume = (volume-dvolume)/volume;
  if (fvolume < 0) error->one(FLERR,"Negative volume found in fix gemc");
  double scale_length = pow(fvolume, 1.0/domain->dimension);

  // DEBUG check that candidate volume move is reasonable
  //printf("%g->%g -> %g\n", volume, dvolume, fvolume);
  //printf("scale_length: %g\n", scale_length);

  // check positions
  //if (myworld == 0) {
  //  printf("%g, %g, %g\n", domain->boxhi[0], domain->boxhi[1], domain->boxhi[2]);
  //  scale_positions(scale_length);
  //}

  //double energy_temp = energy_full();
  //if (mycomm == 0)
  //  printf("energy_now: %g\n", energy_temp);

  // convert to lamda coords so they get scaled
  // TODO : If this is only natom_local, then there is huge spike in energy
  // .... but not with natom_total. So ghost atoms also need to be shifted?
  domain->x2lamda(natom_total);
  for (auto &ifix : rfix) ifix->deform(0);

  // shrink box toward lower corner
  // lower box coordinates always same
  // find center point of box
  xhi_tmp = xlo + Lx*scale_length;
  yhi_tmp = ylo + Ly*scale_length;
  zhi_tmp = zlo + Lz*scale_length;

  // set temporarily
  domain->boxhi[0] = xhi_tmp;
  domain->boxhi[1] = yhi_tmp;
  domain->boxhi[2] = zhi_tmp;

  // reset box and subbox dimensions
  domain->set_global_box();
  domain->set_local_box(); // reassigns sub domains

  // need to migrate after remapping
  auto irregular = new Irregular(lmp);
  irregular->migrate_atoms();
  
  // positions are scaled now
  domain->lamda2x(natom_total);
  for (auto &ifix : rfix) ifix->deform(1);

  // remap call (may lose atoms if no remap)
  // TODO: I think this call is actually not needed since no atoms should pop
  // .... outside the boundes
  domain->remap_all(); // maybe?

  // build neighbor
  // TODO: I think this is necessary, but commenting this out didn't change
  // .... the energy_full(). Could've been just this case.
  neighbor->build(1);

  // check positions
  //if (myworld == 0) {
  //  printf("%g, %g, %g\n", domain->boxhi[0], domain->boxhi[1], domain->boxhi[2]);
  //  scale_positions(scale_length);
  //}

  // check new domain size sync'd
  //printf("%i/%i - dV: %g; scale: %g\n", myworld, mycomm, dvolume, scale_length);
  //printf("xhi: %g, %g, %g\n", xhi_tmp, yhi_tmp, zhi_tmp);
  //error->one(FLERR,"ck");

  // change in potential due to volume change
  // TODO : should include both local and ghost? (is total computed correctly)
  double dU_volume = natom_total*force->boltz*box_temp*log(fvolume);
  // current system energy
  double energy_before = energy_stored;

  //error->one(FLERR,"ck after scale");
  // (possible) future system energy
  double energy_after = energy_full();

  // get total energy from each partition
  double dU;
  if (mycomm == 0) {
    //printf("energy: %g -> %g\n", energy_before, energy_after);
    double idU = energy_after-energy_before-dU_volume;
    // sum change in full energy across each box
    MPI_Allreduce(&idU, &dU, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
  }
  // bcast potential change to rest of world
  MPI_Bcast(&dU, 1, MPI_DOUBLE, 0, world);

  // check potenital change sync'd
  //printf("%i/%i - %g\n", myworld, mycomm, dU);
  //error->one(FLERR,"ck");

  // evaluate probability
  double prob;
  if (dU <= 0.0) prob = 1.0;
  else prob = MIN(exp(-beta*dU),1.0);

  // volume change rejected -> revert atom positions
  if (prob > random_sync->uniform()) {

    //double energy_wrong = energy_full();

    domain->x2lamda(natom_total);
    for (auto &ifix : rfix) ifix->deform(0);

    domain->boxhi[0] = xhi;
    domain->boxhi[1] = yhi;
    domain->boxhi[2] = zhi;

    // reset box and subbox dimensions
    domain->set_global_box();
    domain->set_local_box(); // reassigns sub domains

    irregular->migrate_atoms();
    domain->lamda2x(natom_total);
    for (auto &ifix : rfix) ifix->deform(1);

    // remap call (may lose atoms if no remap)
    domain->remap_all(); // maybe?

    // build neighbor
    neighbor->build(1);

    //printf("%i/%i -- volume fail!\n", myworld, mycomm);

    //double energy_ck = energy_full();
    //printf("should be same: %g - %g; wrong: %g\n",
    //  energy_stored, energy_ck, energy_wrong);
    //error->one(FLERR,"Ck");
  // acccept volume change
  } else {
    nvolume_successes++;
    // store new energy
    energy_stored = energy_after;

    // reacquire upper domain bounds
    xhi = domain->boxhi[0];
    yhi = domain->boxhi[1];
    zhi = domain->boxhi[2];

    // reacquire subdomain bounds
    if (triclinic_flag) {
      sublo = domain->sublo_lamda;
      subhi = domain->subhi_lamda;
    } else {
      sublo = domain->sublo;
      subhi = domain->subhi;
    }
    //printf("%i/%i -- volume success!\n", myworld, mycomm);
  }
  delete irregular;
}

/* ----------------------------------------------------------------------
  Attempt atom exchange
------------------------------------------------------------------------- */
void FixGEMC::attempt_atomic_exchange_full()
{
  nexchange_attempts++;

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

  //printf("%i/%i - sender? %i\n", myworld, mycomm, sender);

  // atom to delete/insert
  int iatom;

  // these are temporarily stored
  double q_iatom;
  int mask_iatom;
  int iatom_type; // record atom type
  double vx,vy,vz;

  q_iatom = 0.0;
  iatom_type = mask_iatom = 0;
  vx = vy = vz = 0.0;
  if (sender) {
    // pick one atom randomly from all atoms in system
    // only one proc will actually delete atom
    update_gas_atoms_list();
    iatom = pick_random_gas_atom();

    //printf("%i/%i - atom? %i\n", myworld, mycomm, iatom);

    double my_q_iatom = 0.0;
    int my_mask_iatom = 0;
    int my_iatom_type = 0;
    double my_vx,my_vy,my_vz;
    my_vx = my_vy = my_vz = 0.0;
    if (iatom >= 0) {
      mask_iatom = atom->mask[iatom]; // store particle mask
      my_iatom_type = atom->type[iatom]; // for insertion
      atom->mask[iatom] = exclusion_group_bit;
      // check if charged
      if (q_flag) {
        q_iatom = atom->q[iatom];
        atom->q[iatom] = 0.0;
      }

      // store velocity
      my_vx = atom->v[iatom][0];
      my_vy = atom->v[iatom][1];
      my_vz = atom->v[iatom][2];

      if (force->kspace) force->kspace->qsum_qsq();
      if (force->pair->tail_flag) force->pair->reinit();
    }

    //printf("%i/%i - atom_type? %i\n", myworld, mycomm, my_iatom_type);
    //printf("%i/%i - v? %g,%g,%g\n",
    //  myworld, mycomm, my_vx, my_vy, my_vz);

    // have comm 0 reduce since it may not be the one deleting
    MPI_Reduce(&my_iatom_type, &iatom_type, 1, MPI_INT, MPI_SUM, 0, world);
    MPI_Reduce(&my_mask_iatom, &mask_iatom, 1, MPI_INT, MPI_SUM, 0, world);
    MPI_Reduce(&my_q_iatom,    &q_iatom,    1, MPI_DOUBLE, MPI_SUM, 0, world);
    MPI_Reduce(&my_vx, &vx, 1, MPI_DOUBLE, MPI_SUM, 0, world);
    MPI_Reduce(&my_vy, &vy, 1, MPI_DOUBLE, MPI_SUM, 0, world);
    MPI_Reduce(&my_vz, &vz, 1, MPI_DOUBLE, MPI_SUM, 0, world);

    //printf("%i/%i - atom_type? %i\n", myworld, mycomm, iatom_type);
    //printf("%i/%i - v? %g,%g,%g\n",
    //  myworld, mycomm, vx,vy,vz);
  }

  //energy_after = energy_full();
  //printf("%i/%i - e after? %g\n", myworld, mycomm, energy_after);

  // tell other box the atome type, mask, charge, and velocity it's receiving
  int all_iatom_type, all_mask_iatom;
  double all_q_iatom;
  double all_vx, all_vy, all_vz;
  if (mycomm == 0) {
    MPI_Allreduce(&iatom_type, &all_iatom_type, 1, MPI_INT, MPI_SUM, comm_replica);
    MPI_Allreduce(&mask_iatom, &all_mask_iatom, 1, MPI_INT, MPI_SUM, comm_replica);
    MPI_Allreduce(&q_iatom,    &all_q_iatom, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
    MPI_Allreduce(&vx, &all_vx, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
    MPI_Allreduce(&vy, &all_vy, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
    MPI_Allreduce(&vz, &all_vz, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
  }

  // tell all other procs the atom props
  MPI_Bcast(&all_iatom_type, 1, MPI_INT, 0, world);
  MPI_Bcast(&all_mask_iatom, 1, MPI_INT, 0, world);
  MPI_Bcast(&all_q_iatom, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&all_vx, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&all_vy, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&all_vz, 1, MPI_DOUBLE, 0, world);

  //printf("%i/%i - t %i m %i v? %g,%g,%g\n",
  //  myworld, mycomm, all_iatom_type, all_mask_iatom,
  //  all_vx, all_vy, all_vz);
  //error->one(FLERR,"ck");

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
      atom->avec->create_atom(all_iatom_type,coord);
      int jatom = atom->nlocal - 1;

      // add to groups
      // optionally add to type-based groups

      atom->mask[jatom] = all_mask_iatom;
      atom->v[jatom][0] = all_vx;
      atom->v[jatom][1] = all_vy;
      atom->v[jatom][2] = all_vz;
      if (q_flag) atom->q[jatom] = all_q_iatom;
      modify->create_attribute(jatom);
    } // END if proc_flag

    atom->natoms++;

    // TODO: What is happening here
    // if tag's enabled, mapping local to global ids
    if (atom->tag_enable) {
      atom->tag_extend();
      // cctually mapping
      if (atom->map_style != Atom::MAP_NONE) atom->map_init();
    }

    //atom->nghost = 0; // probably useless
    if (triclinic_flag) domain->x2lamda(atom->nlocal);
    comm->borders();
    if (triclinic_flag) domain->lamda2x(atom->nlocal+atom->nghost);
    if (force->kspace) force->kspace->qsum_qsq();
    if (force->pair->tail_flag) force->pair->reinit();
  } // END if sender

  // everyone call full_energy
  double energy_before = energy_stored;
  double energy_after = energy_full();

  // evalute probability for exchange
  double dU;
  int success;
  if (comm->me == 0) {
    double idU = energy_after-energy_before;
    MPI_Allreduce(&idU,&dU,1,MPI_DOUBLE,MPI_SUM,comm_replica);
    double volume = (xhi-xlo)*(yhi-ylo)*(zhi-zlo);
    double NV;
    if (sender) NV = volume/atom->natoms;
    else NV = (atom->natoms)/volume;
    double allNV;
    MPI_Allreduce(&NV,&allNV,1,MPI_DOUBLE,MPI_PROD,comm_replica);

    dU += (box_temp*force->boltz*log(allNV));
    double prob = MIN(exp(-beta*dU),1.0);
    if (prob >= random->uniform()) success = 1;
    MPI_Bcast(&success, 1, MPI_INT, 0, comm_replica);
  }
  MPI_Bcast(&success, 1, MPI_INT, 0, world);

  //printf("%i/%i - success? %i\n", myworld, mycomm, success);

  if (sender) {
    // delete iatom
    if (success) {
      nexchange_successes++;
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
        atom->mask[iatom] = all_mask_iatom;
        if (q_flag) atom->q[iatom] = all_q_iatom;
      }
      if (force->kspace) force->kspace->qsum_qsq();
      if (force->pair->tail_flag) force->pair->reinit();
      energy_stored = energy_before;
    }
  } else {
    // accept newly inserted iatomthere
    if (success) {
      if (mycomm == 0) printf("success!\n");
      nexchange_successes++;
      energy_stored = energy_after;
    // remove newly inserted iatom
    } else {
      if (mycomm == 0) printf("fail!\n");
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
  ntranslation_attempts++;

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
  //if (myworld == 0)
  //  printf("%i -> %i :: %i : %4.3e -> %4.3e\n",
  //    myworld, mycomm, i, energy_before, energy_after);

  // DEBUG: Check if RNG sync'd 
  //for (int r = 0; r < 8; r++) random_sync->uniform();
  //for (int r = 0; r < 12; r++) random->uniform();
  //printf("%i, %i rng: %g - %g\n",
  //  myworld, mycomm, random_sync->uniform(), random->uniform());

  double prob;
  if (energy_after <= energy_before) prob = 1.0;
  else prob = exp(beta*(energy_before - energy_after));

  //if (myworld == 0)
  //  printf("%g - %g -> %g - %g\n", energy_before, energy_after,
  //    prob, random_sync->uniform());

  // TODO : More thorough testing that successes and fails sync'd
  if (energy_after < MAXENERGYTEST &&
      random_sync->uniform() < prob) {
    energy_stored = energy_after;
    //if (myworld == 0) printf("%i - failure!\n", mycomm);
  } else {
    ntranslation_successes++;
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
    //if (myworld == 0) printf("%i - success!\n", mycomm);
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
      random_sync->uniform() <
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

/* ----------------------------------------------------------------------
  Scale new particle positions according to volume change
------------------------------------------------------------------------- */
void FixGEMC::scale_positions(const double scale)
{
  double **x = atom->x;
  printf("%g, %g, %g\n",
    x[1][0], x[1][1], x[1][2]);

}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixGEMC::pick_random_gas_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (natom_total*random_sync->uniform());
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
  int iwhichglobal = static_cast<int> (natom_local*random_sync->uniform());
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

