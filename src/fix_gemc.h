/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(gemc,FixGEMC);
// clang-format on
#else

#ifndef LMP_FIX_GEMC_H
#define LMP_FIX_GEMC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixGEMC : public Fix {
 public:
  FixGEMC(class LAMMPS *, int, char **);
  ~FixGEMC() override;
  int setmask() override;
  void init() override;
  void pre_exchange() override;

 private:

  // vars //

  // user provided inputs
  
  int nevery; // frequency this fix is called
  int ntranslate; // number of translation each box performs each step
  int nrotate; // number of rotations each box performs each setp
  int nexchange; // number of particle exchanges between the boxes each step
  int nvolume; // number of volume exchanges between the boxes each step
  double box_temp; // temperature of each box (assumed equal)
  double displace; // maximum displacement for traslations
  double max_volume; // maximum volume change requested
  double min_box_volume; // minimum box volume
  int seed; // RNG seed

  // for evaluating probability

  int overlap_flag; // check for overlap?
  double overlap_cutoffsq; // check for max cutoff 
  double beta; // 1 / (boltzmann * temperature)
  double energy_stored; // current potential energy
  class Compute *c_pe; // compute to get full potential energy

  // for determining which move to make

  int nmoves; // total MC moves (translate/rotate + exchange + volume)
  // cummulative probabilites
  double pc_exchange; // probability MC move is an exchange
  double pc_volume; // probability MV move is a volume change
  double pc_translate; // probability MC move is a translation
  double pc_rotate; // probability MC move is a rotation

  // for tracking how many attempts/successes

  int ntranslation_attempts;
  int ntranslation_successes;
  int nrotation_attempts;
  int nrotation_successes;
  int nexchange_attempts;
  int nexchange_successes;
  int nvolume_attempts;
  int nvolume_successes;

  // particle - related props

  int natom_lower; // lower index for local atoms
  int natom_local; // number of atoms in this proc
  int natom_total; // total number of atoms in world I'm in

  int molecule_flag; // 0 for atom; 1 for molecule
  int full_flag; // compute full energy
  int q_flag; // particles charged?

  // MC exchange

  int exclusion_group, exclusion_group_bit; // mask for excluding certain atoms

  // domain - related props

  int triclinic_flag;
  double xlo, ylo, zlo; // lower domain bounds
  double xhi, yhi, zhi; // upper domain bounds
  double *sublo, *subhi; // sub domain bounds
  double xhi_tmp, yhi_tmp, zhi_tmp; // temporary upper domain bounds
  std::vector<Fix *> rfix;    // indices of rigid fixes

  // for communication

  int myworld;
  int mycomm; // rank in my world
  int nprocs;
  int myrank_replica; // rand between boxes

  double *commbuf;
  MPI_Comm comm_replica; // for communication between partitions

  class RanPark *random_universe; // sync'd RNG for all worlds
  class RanPark *random_world; // sync'd RNG for one world
  class RanPark *random_proc; // RNG for each proc (not sync'd)

  // additional comm (mostly for exchange)

  int maxbuf;           // size of buf send/recv in # of doubles
  double *buf;    // bufs used in migrate_atoms

  // misc
  int progress; // tracks remaining simulation time

  // subroutines //

  // optional args that user can provide

  void options(int, char **);

  // for MC translate/rotation moves

  void attempt_atomic_translation_full();
  void attempt_molecule_translation_full();
  void attempt_molecule_rotation_full();

  // for MC volume moves (always full)

  void attempt_volume_change_full();
  void scale_positions(const double);
  void unscale_positions(const double);

  // for MC exchange moves

  void attempt_atomic_exchange_full();
  void attempt_molecule_exchange_full();

  int init_exchange();
  void grow_sendrecv();

  // misc functions for all MC moves

  double energy_full(); // computes full potential energy
  void update_gas_atoms_list(); // updates count for local number of atoms
  int pick_random_gas_atom(); // picks random atom
  tagint pick_random_gas_molecule(); // picks random atom

};

}    // namespace LAMMPS_NS

#endif
#endif
