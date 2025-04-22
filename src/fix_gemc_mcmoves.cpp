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
// TODO : got error
//   "Neighbor list overflow, boost neigh_modify one (../npair_bin.cpp:248)"
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

    // make sure volume change less than available volume
    while (1) {
      dvolume = (random_proc->uniform()-0.5)*max_volume;
      if ((min_volume+dvolume) > min_box_volume) break;
    }

    double sum_dvolume;
    MPI_Allreduce(&dvolume, &sum_dvolume, 1, MPI_DOUBLE, MPI_SUM, comm_replica);

    // set the volume change as average    
    double avg_dvolume = sum_dvolume*0.5;

    // have one world invert the volume change so total volume conserved
    if (dvolume < avg_dvolume) dvolume = -avg_dvolume;
    else dvolume = avg_dvolume;
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
  // because relative positions are the same in a volume move
  // neighbor lists do exchanges anyways
  // would not need this call because no shape change
  //auto irregular = new Irregular(lmp);
  //irregular->migrate_atoms();

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
  //neighbor->build(1);

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
  if (prob > random_universe->uniform()) {

    //double energy_wrong = energy_full();

    domain->x2lamda(natom_total);
    for (auto &ifix : rfix) ifix->deform(0);

    domain->boxhi[0] = xhi;
    domain->boxhi[1] = yhi;
    domain->boxhi[2] = zhi;

    // reset box and subbox dimensions
    domain->set_global_box();
    domain->set_local_box(); // reassigns sub domains

    //irregular->migrate_atoms();
    domain->lamda2x(natom_total);
    for (auto &ifix : rfix) ifix->deform(1);

    // remap call (may lose atoms if no remap)
    domain->remap_all(); // maybe?

    // build neighbor
    neighbor->build(1);

    Lx = xhi-xlo;
    Ly = yhi-ylo;
    Lz = zhi-zlo;
    volume = Lx*Ly*Lz;

    //printf("%i/%i -- volume fail! - %g, %g\n", myworld, mycomm, dvolume, volume);

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

    //printf("%i/%i -- volume success! -- %g - %g\n", myworld, mycomm, dvolume, volume);
  }
  //delete irregular;
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
    double drand = random_proc->uniform();
    double dmean;
    MPI_Allreduce(&drand, &dmean, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
    dmean *= 0.5;
    if (drand > dmean) sender = 1;
    else sender = 0;
  }
  MPI_Bcast(&sender, 1, MPI_INT, 0, world);

  // check roles are sync'd
  //if (myworld == 0)
  //  printf("%i/%i - sender? %i\n", myworld, mycomm, sender);
  //error->one(FLERR,"ck");

  //double energy_ck = energy_full();
  //printf("should be same %i/%i - %g -> %g\n",
  //  myworld, mycomm, energy_stored, energy_ck);

  // set up for atom exchange
  init_exchange();



  int nsend;
  int send_comm;

  // atom to delete/insert
  int iatom;

  // these are temporarily stored

  if (sender) {
    // pick one atom randomly from all atoms in system
    // only one proc will actually delete atom
    iatom = pick_random_gas_atom();

    //printf("%i/%i - atom? %i\n", myworld, mycomm, iatom);

    // have associated proc pack atom
    if (iatom >= 0) {
      nsend = atom->avec->pack_exchange(iatom,&buf_send);
      // not sure if these are needed
      //if (force->kspace) force->kspace->qsum_qsq();
      //if (force->pair->tail_flag) force->pair->reinit();

      // send to proc 0 if it doesn't already have it
      if (mycomm != 0)
        MPI_Send(&buf_send, nsend, MPI_DOUBLE, 0, 0, world);
    }

    //int my_pair[2];
    //my_pair[0] = iatom;
    //my_pair[1] = mycomm;

    // find which proces from each box to pair
    //int max_pair[2];
    //MPI_Allreduce(my_pair, max_pair, 1, MPI_2INT, MPI_MAXLOC, world);
    //int send_comm = max_pair[1];

    // have send comm bcast info
    //MPI_Bcast(&q_iatom, 1, MPI_DOUBLE, send_comm, world);
    //MPI_Bcast(&type_iatom, 1, MPI_INT, send_comm, world);
    //MPI_Bcast(&mask_iatom, 1, MPI_INT, send_comm, world);
    //MPI_Bcast(&vx, 1, MPI_DOUBLE, send_comm, world);
    //MPI_Bcast(&vy, 1, MPI_DOUBLE, send_comm, world);
    //MPI_Bcast(&vz, 1, MPI_DOUBLE, send_comm, world);

    //printf("%i - %i max_pair: %i -> %i %i\n",
    //  myworld, mycomm, iatom, max_pair[0], max_pair[1]);

    // at this point, one of the procs in the box has deleted.
    // all procs

    //printf("%i/%i - atom_type? %i\n", myworld, mycomm, type_iatom);
    //printf("%i/%i - v? %g,%g,%g\n",
    //  myworld, mycomm, vx, vy, vz);

    // have comm 0 reduce since it may not be the one deleting
    //MPI_Reduce(&my_iatom_type, &iatom_type, 1, MPI_INT, MPI_SUM, 0, world);
    //MPI_Reduce(&my_mask_iatom, &mask_iatom, 1, MPI_INT, MPI_SUM, 0, world);
    //MPI_Reduce(&my_q_iatom,    &q_iatom,    1, MPI_DOUBLE, MPI_SUM, 0, world);
    //MPI_Reduce(&my_vx, &vx, 1, MPI_DOUBLE, MPI_SUM, 0, world);
    //MPI_Reduce(&my_vy, &vy, 1, MPI_DOUBLE, MPI_SUM, 0, world);
    //MPI_Reduce(&my_vz, &vz, 1, MPI_DOUBLE, MPI_SUM, 0, world);

    //printf("%i/%i - atom_type? %i\n", myworld, mycomm, iatom_type);
    //printf("%i/%i - v? %g,%g,%g\n",
    //  myworld, mycomm, vx,vy,vz);
  }

  //energy_after = energy_full();
  //printf("%i/%i - e after? %g\n", myworld, mycomm, energy_after);

  // tell other box the atome type, mask, charge, and velocity it's receiving
  // there could be a lot more. pack exchange within atom_vec

  /*

  int recv_iatom_type, recv_mask_iatom;
  double recv_q_iatom;
  double recv_vx, recv_vy, recv_vz;
  if (mycomm == 0) {
    if (sender) {
      MPI_Allreduce(&iatom_type, &all_iatom_type, 1, MPI_INT, MPI_SUM, comm_replica);
      MPI_Allreduce(&mask_iatom, &all_mask_iatom, 1, MPI_INT, MPI_SUM, comm_replica);
      MPI_Allreduce(&q_iatom,    &all_q_iatom, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
      MPI_Allreduce(&vx, &all_vx, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
      MPI_Allreduce(&vy, &all_vy, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
      MPI_Allreduce(&vz, &all_vz, 1, MPI_DOUBLE, MPI_SUM, comm_replica);
    } else {

    }
  }

  // tell all other procs the atom props
  MPI_Bcast(&all_iatom_type, 1, MPI_INT, 0, world);
  MPI_Bcast(&all_mask_iatom, 1, MPI_INT, 0, world);
  MPI_Bcast(&all_q_iatom, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&all_vx, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&all_vy, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&all_vz, 1, MPI_DOUBLE, 0, world);

  */

  // check everyone has same atom being exchanged
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
        lamda[0] = random_proc->uniform();
        lamda[1] = random_proc->uniform();
        lamda[2] = random_proc->uniform();

        // wasteful, but necessary

        if (lamda[0] == 1.0) lamda[0] = 0.0;
        if (lamda[1] == 1.0) lamda[1] = 0.0;
        if (lamda[2] == 1.0) lamda[2] = 0.0;

        domain->lamda2x(lamda,coord);
      } else {
        coord[0] = xlo + random_proc->uniform() * (xhi-xlo);
        coord[1] = ylo + random_proc->uniform() * (yhi-ylo);
        coord[2] = zlo + random_proc->uniform() * (zhi-zlo);
      }
    } // END mycomm

    MPI_Bcast(&coord, 3, MPI_DOUBLE, 0, world);

    //printf("%i/%i - %g, %g, %g\n",
    //  myworld, mycomm, coord[0], coord[1], coord[2]);

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
      // unpack
      atom->avec->create_atom(type_iatom,coord);
      int jatom = atom->nlocal - 1;

      // add to groups
      // optionally add to type-based groups

      atom->mask[jatom] = mask_iatom;
      atom->v[jatom][0] = vx;
      atom->v[jatom][1] = vy;
      atom->v[jatom][2] = vz;
      if (q_flag) atom->q[jatom] = q_iatom;
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
    if (triclinic_flag) domain->x2lamda(natom_local);
    comm->borders();
    if (triclinic_flag) domain->lamda2x(natom_total);
    if (force->kspace) force->kspace->qsum_qsq();
    if (force->pair->tail_flag) force->pair->reinit();
  } // END if sender

  // everyone call full_energy
  double energy_before = energy_stored;
  double energy_after = energy_full();

  // evalute probability for exchange
  int success;
  if (mycomm == 0) {
    double dU;
    double idU = energy_after-energy_before;
    //printf("%i/%i - %g -> %g\n", myworld, mycomm, energy_before, energy_after);
    MPI_Allreduce(&idU,&dU,1,MPI_DOUBLE,MPI_SUM,comm_replica);
    //printf("%i/%i - %g -> %g\n", myworld, mycomm, idU, dU);

    double volume = (xhi-xlo)*(yhi-ylo)*(zhi-zlo);
    double NV;
    // TODO : I believe the send should be over current natoms - 1
    // ... does it included count if it has exclusion group bit?
    if (sender) NV = volume/(atom->natoms-1);
    else NV = atom->natoms/volume;
    double allNV;
    MPI_Allreduce(&NV,&allNV,1,MPI_DOUBLE,MPI_PROD,comm_replica);

    //printf("%i/%i - %g , %g-> %g\n", myworld, mycomm, volume, NV, allNV);

    dU += (box_temp*force->boltz*log(allNV));
    double prob = MIN(exp(-beta*dU),1.0);

    if (prob > random_proc->uniform()) success = 1;
    else success = 0;

    MPI_Bcast(&success, 1, MPI_INT, 0, comm_replica);

    //if (myworld == 0)
    //  printf("%i/%i - prob: %g -> %g; success? %i\n",
    //    myworld, mycomm, beta*dU, prob, success);

  }
  MPI_Bcast(&success, 1, MPI_INT, 0, world);

  //if (myworld == 0)
  //  printf("%i/%i - success? %i\n", myworld, mycomm, success);
  //error->one(FLERR,"ck");

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
      //printf("%i/%i - deleted!\n", myworld, mycomm);
    // revert iatom (do not delete)
    } else {
      if (iatom >= 0) {
        atom->mask[iatom] = mask_iatom;
        if (q_flag) atom->q[iatom] = q_iatom;
      }
      if (force->kspace) force->kspace->qsum_qsq();
      if (force->pair->tail_flag) force->pair->reinit();
      energy_stored = energy_before;
      //printf("%i/%i - fail deleted!\n", myworld, mycomm);
    }
  } else {
    // accept newly inserted iatomthere
    if (success) {
      nexchange_successes++;
      energy_stored = energy_after;
      //printf("%i/%i - stored!\n", myworld, mycomm);
    // remove newly inserted iatom
    } else {
      atom->natoms--;
      if (proc_flag) atom->nlocal--;
      if (force->kspace) force->kspace->qsum_qsq();
      if (force->pair->tail_flag) force->pair->reinit();
      energy_stored = energy_before;
      //printf("%i/%i - fail stored!\n", myworld, mycomm);
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

  if (natom_total == 0) return;

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
      rx = 2*random_proc->uniform() - 1.0;
      ry = 2*random_proc->uniform() - 1.0;
      rz = 2*random_proc->uniform() - 1.0;
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

  double prob;
  if (energy_after <= energy_before) prob = 1.0;
  else prob = exp(beta*(energy_before - energy_after));

  // TODO : More thorough testing that successes and fails sync'd
  if (energy_after < MAXENERGYTEST &&
      random_world->uniform() < prob) {
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

int FixGEMC::pick_random_gas_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (natom_total*random_world->uniform());
  if ((iwhichglobal >= natom_lower) &&
      (iwhichglobal < natom_lower + natom_local)) {
    i = iwhichglobal - natom_lower;
  }

  return i;
}

