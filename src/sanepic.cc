#include <cmath>
#include <fcntl.h>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
// #include "nag.h"
// #include "nagf04.h"
// #include "nagg05.h"
#include "mpi.h"
#include <cstring>

extern "C" {
#include <fitsio.h>
}

extern "C" {
void GetProcMem(long *vmem, long *phymem);
}

using namespace std;

#define NR_END 1
#define D2PI 6.2831853071795864769252867665590057683943387987502
#define dmod(A, B)                                                             \
  ((B) != 0.0 ? ((A) * (B) > 0.0 ? (A) - (B)*floor((A) / (B))                  \
                                 : (A) + (B)*floor(-(A) / (B)))                \
              : (A))

#define EXITMESS                                                               \
  { fprintf(stderr, "%s %d\n", __FILE__, __LINE__); }
#define exit_Sanepic(MyErr)                                                    \
  { MPI_Abort(MPI_COMM_WORLD, (int)MyErr); }

long *data_compare;

int compare_global_array_long(const void *a, const void *b) {
  const long *da = (const long *)a;
  const long *db = (const long *)b;
  return (data_compare[*da] > data_compare[*db]) -
         (data_compare[*da] < data_compare[*db]);
}

int compare_long(const void *a, const void *b) {
  const long *da = (const long *)a;
  const long *db = (const long *)b;
  return (*da > *db) - (*da < *db);
}

void do_PtNd(double *PNd, double *sumCross, fftw_complex *fdatas,
             long *datatopix, double *psip, double *SpN_all, long ns, long ndet,
             long *indpix, long ipix_min, long npix, bool CORRon, long iseg,
             long iseg_min, long nseg, unsigned char *segon, long lnr,
             double *relCalib, double *Mp, long *hits, bool polar,
             double *polardet, double *polarang, bool GalBP, double *relfGal,
             int idetr);

void crosspar(double *PNd, long *datatopix, double *psip, long ns, long ndet,
              long *indpix, long ipix_min, long npix, long iseg, long iseg_min,
              long nseg, unsigned char *segon, long lnr, bool polar,
              double *polardet, double *polarang, int idetr);

void compute_ftrProcesdata(double *datas, fftw_complex *fdatas, long ns,
                           long ndet, long iseg, long nseg, long idetr);

void write_tfAS(double *S, fftw_complex *fdatas, long *datatopix, long *indpix,
                double *psip, long npix, long ns, long ndet, long iseg,
                long iseg_min, long nseg, unsigned char *segon, long lnr,
                bool polar, double *polardet, double *polarang, int GalBP,
                double *relfGal, int idetr);

void deproject(double *S, long *indpix, long *datatopix, double *psip, long ns,
               long npix, double *Ps, bool polar, double polard, double polara,
               int GalBP, double rfGal);

void deproject_partial(double *S, long *indpix, long *datatopix, long ns,
                       double *Ps);

void compute_Nm1d(double *d, long ns, double *SpN_all, double relcal,
                  double *Nm1d);

void compute_diagPtNPCorr(double *Nk, long *datatopix, long ns, long *indpix,
                          long ipix_min, int npix, double *dPtNP);

void ang2pix_ring(const long nside, double theta, double phi, long *ipix);

void write_reduced_map(char *mapoutstr, long nside, double *mapr, long *indpix);

void fMPI_merge(double *lmap, double *d, long nsmpix, long npix, long npixtot,
                int rank, int newrank, int size, int nsubm, bool polar,
                bool GalBP, MPI_Status status, bool allprc, MPI_Comm newcomm);

long WriteMAP(void *map1d, char *mapoutstr, int type, long nn);

long ReadLongVECT(long *data, string filename, long imin, long imax);

long ReadDoubleVECT(double *data, string filename, long imin, long imax);

long ReadVECT(void *data, string filename, long typesize, long imin, long imax);

void read_bolofile(string fname, list<string> &bolos);

void fmodelTC(double *pp, long ns, fftw_complex *Fu);

double MPI_compute_chi2(fftw_complex *fdatas0, fftw_complex *fodipole,
                        fftw_complex *fdatas, fftw_complex *fdata, double *map,
                        long nseg, long *datatopix, long *indpix, double *psip,
                        double *relCalib, unsigned char *segon, double *SpN_all,
                        long npix, long ns, long ndet, long iseg_min, long lnr,
                        int nsubm, bool polar, double *polardet,
                        double *polarang, bool GalBP, double *relfGal,
                        long newrank);

void forceLeakzero(double *PtNPmatStot, long nside, long *indpix, long ipixmin,
                   long nsmpix, bool polar);

void zeros(double *data, long nn);

void initarray(double *data, long nn, double val);

void read_bolo_offsets(string bolo, string file_BoloOffsets, double *offsets);

void slaDs2tp(double ra, double dec, double raz, double decz, double *xi,
              double *eta, int *j);

void slaDtp2s(double xi, double eta, double raz, double decz, double *ra,
              double *dec);

double slaDranrm(double angle);

void pointingshift(double x, double y, long nn, double *thetap, double *phip,
                   double *psip, double *theta, double *phi, double *psi);

template <class T> void list2array(list<T> l, T *a) {

  // copy list of type T to array of type T
  typename list<T>::iterator iter;
  int i;

  for (iter = l.begin(), i = 0; iter != l.end(); iter++, i++) {
    a[i] = *iter;
  }
}

using namespace std;

void usage(char *name) {
  cerr << "USAGE: " << name << ": Map-making" << endl << endl;
  cerr << name << " [-F <data dir>] [-f <used samples (file name)>]" << endl;
  cerr << "-H <filter freq>       frequency of the high pass filter applied to "
          "the data"
       << endl;
  cerr << "-J <noise freq cut>    frequency under which noise power spectra "
          "are thresholded. Default is the frequency cut of the high pass "
          "filter applied to the data (-H option)"
       << endl;
  cerr << "\t[-O <output file>] [-B <bolo ext>] [-G <flag ext>] [-P <pointing "
          "ext>]"
       << endl;
  cerr << "\t[-o <orbital dipole on>] [-e <output file>]" << endl << endl;
  cerr << "-A <apodize nsamp>     number of samples to apodize (supercedes "
          "-a)\n";
  cerr << "-m <padding interval>  number of samples extrapolated before (and "
          "after) data"
       << endl;
  cerr << "-B <bolo ext>          bolo field extension"
       << endl; // Detector name $dataSuffixe
  cerr << "-d <Det number>        Number of detectors to include" << endl;
  cerr << "-D <detector list>     List of detectors to include" << endl;
  cerr << "-e <output file str>   id string in output file" << endl;
  cerr << "-F <data dir>          source data directory"
       << endl; // input data directory for toi
  cerr << "-Z <pointing dir>      source pointing directory + prefixe" << endl;
  cerr << "-f <sample list file mane>  file including list of 1st and last "
          "samples"
       << endl;
  cerr << "-G <flag ext>          flag field extension" << endl;
  cerr << "-H <filter freq>       filter frequency (multiple allowed)" << endl;
  cerr << "-k <noise prefixe>     noise power spectrum file prefixe " << endl;
  cerr << "-u Noise Nb samp       Number of samples of the noise power spectra"
       << endl;
  cerr << "-K <File Leak coef>    Name of file containing the coefficients of "
          "the Leakage component"
       << endl;
  cerr << "-X <det location file> File providing pointing info for each "
          "detector wrt boresight"
       << endl;
  cerr << "-O <output dir>        output directory" << endl;
  cerr << "-o <orbital dipole>    consider orbital dipole" << endl;
  cerr << "-P <pointing ext>      pointing field extension" << endl;
  cerr << "-p <polar>             set this keyword if polarization is included"
       << endl;
  cerr << "-b <extra component>   set this keyword if extra component is "
          "estimated"
       << endl;
  cerr << "-l <data seg length>   Lenght of data segments. Must be the same "
          "for all segments"
       << endl;
  cerr << "-n <nb of segments>    Number of data segments" << endl;
  cerr << "-N <Nside>             Healpix nside" << endl;
  cerr << "-R <Global recalib>    Global recalibration coef for each detector"
       << endl;
  cerr << "-r <recalib>           If set, recalibrate data on some timescale "
          "(to implement)"
       << endl;
  cerr << "-c <recalib leakage>   If set, recalibrate leakage component"
       << endl;
  cerr << "-j <Delt iter>         Number of iterations after which the maps "
          "are recalculated"
       << endl;
  cerr << "-L <no baseline>       Keyword specifiying if a baseline is removed "
          "from the data or not (0 for YES, 1 for NO)"
       << endl;
  cerr << "-z <precompute PNd>    Keyword specifying if PNd is precomputed and "
          "read from disk"
       << endl;
  cerr << "-g <project gaps>      Keyword specifying if gaps are projected to "
          "a pixel in the map, if so gap filling of noise only is performed "
          "iteratively. Default is 0"
       << endl;
  cerr << "-M <map flagged data>   Keyword specifying if flagged data are put "
          "in a separate map, default is 0"
       << endl;
  cerr << "-E <no corr>           Set this keyword to 0 if correlations are "
          "not included in the analysis"
       << endl;
  cerr << "-h <HWP on>            Set if Half Wave Plate. Default is 0" << endl;
  exit(1);
}

//**********************************************************************************//
//**********************************************************************************//
//*************************** Beginning of main program
//****************************//
//**********************************************************************************//
//**********************************************************************************//

int main(int argc, char *argv[]) {

  int MyErr;
  long NbWrite;
  string Command, CommandW, MyStr;
  char mapoutstr[100], NoiseSpFiledet[100], namefilering[300];
  string pbrGRP, pbrptgGRP, vectGRP, logger, logStr;

  int size;
  int rank;
  MPI_Status status;

  // setup MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
    printf("%d\n", __LINE__);

  long ii, jj, kk, ll, iseg, idet, iseg_min, iseg_max, ipr;
  double xsh, ysh;
  int iterw = 10;
  bool CORRon = 0, polar = 0, GalBP = 0, recalLeak = 0, recalib = 0,
       applycalib = 0, recalibGlobal = 0;
  bool odip = 0, recalPolar = 0, Destrip = 0;
  long ndata, nsegtot, nseg, ndet, nside, npix, npixtot, ns, nssp;
  double theta, phi;
  double *datas, *ra, *dec, *psip, *psipFP, *odipole;
  double *PNd, *PNdtot, *SpN_all = NULL, *SpN = NULL, *polardet = NULL,
                        *polarang = NULL, *relfGal = NULL;
  long *isampmin, *isampmax;
  double *offx, *offy, *psi;
  long *indpix, *pixon, *pixonall, *datatopix;
  fftw_complex *fodipole, *fdatas, *fdata, *fdatas0;

  recalPolar = 0; //// Part of code to check

  bool WriteCrossLink = 1;

  int retval;
  double f_lp, f_lp_Nk, OmegaHWP;
  long napod, marge;
  bool PNd_ready = 0;
  bool flgdupl = 0;
  bool NORMLIN = 0;
  bool projgaps = 0;
  bool pfile = 0;
  bool initMapOn = 0;
  bool HWP = 0;
  bool writeIterOn = 1;
  string outdir = "";
  string bextension = "";
  string fextension = "";
  string pextension = "";
  string termin_out = "";
  string noiseSppreffile = "";
  string RelfGal;
  string filesegnum;
  string dirfile = "";
  string dirfilep = "";
  string initMap = "";
  string recalibLeak = "";
  string filedetname;
  string filedetinfo = "";
  list<string> bolos;

  // char ffname[100];
  // strcpy(ffname,fname.c_str());

  // Parse command line options
  while ((retval = getopt(argc, argv,
                          "F:Z:f:n:l:d:D:C:H:h:w:J:o:O:B:R:X:G:P:e:p:A:m:k:K:u:"
                          "c:N:n:L:g:r:M:E:j:z:b:I:i:")) != -1) {
    switch (retval) {
    case 'F':
      dirfile = optarg;
      break;
    case 'Z':
      dirfilep = optarg;
      pfile = 1;
      break;
    case 'f':
      filesegnum = optarg;
      break;
    case 'p':
      polar = atoi(optarg);
      break;
    case 'b':
      GalBP = atoi(optarg);
      break;
    case 'H':
      f_lp = atof(optarg);
      break;
    case 'J':
      f_lp_Nk = atof(optarg);
      break;
    case 'A':
      napod = atoi(optarg);
      break;
    case 'm':
      marge = atoi(optarg);
      break;
    case 'd':
      ndet = atoi(optarg);
      break;
    case 'D':
      read_bolofile(optarg, bolos);
      cerr << "num bolos: " << bolos.size() << endl;
      break;
    case 'C':
      bolos.push_back(optarg);
      break;
    case 'O':
      outdir = optarg;
      break;
    case 'B':
      bextension = optarg;
      break;
    case 'R':
      recalibGlobal = atoi(optarg);
      break;
    case 'X':
      filedetinfo = optarg;
      break;
    case 'G':
      fextension = optarg;
      break;
    case 'P':
      pextension = optarg;
      break;
    case 'o':
      odip = atoi(optarg);
      break;
    case 'e':
      termin_out = optarg;
      break;
    case 'k':
      noiseSppreffile = optarg;
      break;
    case 'u':
      nssp = atoi(optarg);
      break;
    case 'r':
      recalib = atoi(optarg);
      break;
    case 'c':
      recalibLeak = atoi(optarg);
      break;
    case 'N':
      nside = atoi(optarg);
      break;
    case 'n':
      nsegtot = atoi(optarg);
      break;
    case 'l':
      ns = atoi(optarg);
      break;
    case 'L':
      NORMLIN = atoi(optarg);
      break;
    case 'g':
      projgaps = atoi(optarg);
      break;
    case 'z':
      PNd_ready = atoi(optarg);
      break;
    case 'M':
      flgdupl = atoi(optarg);
      break;
    case 'E':
      CORRon = atoi(optarg);
      break;
    case 'j':
      iterw = atoi(optarg);
      break;
    case 'K':
      RelfGal = optarg;
      break;
    case 'h':
      HWP = atoi(optarg);
      break;
    case 'w':
      OmegaHWP = atof(optarg);
      break;
    case 'I':
      initMap = optarg;
      initMapOn = 1;
      break;
    case 'i':
      writeIterOn = atoi(optarg);
      break;
    default:
      cerr << "Option '" << (char)retval << "' not valid. Exiting.\n\n";
      usage(argv[0]);
    }
  }

  if (pfile == 0)
    dirfilep = dirfile.c_str();

  if (initMapOn == 0)
    initMap = initMap.c_str();

  string *bolonames;
  bolonames = new string[ndet];
  list2array(bolos, bolonames);

  // Read pointing info file
  double *offsets;
  offsets = new double[3];
  offx = new double[ndet];
  offy = new double[ndet];
  psi = new double[ndet];
  for (idet = 0; idet < ndet; idet++) {
    read_bolo_offsets(bolonames[idet], filedetinfo, offsets);
    offx[idet] = offsets[1] / 180.0 * D2PI / 2.0;
    offy[idet] = offsets[0] / 180.0 * D2PI / 2.0;
    psi[idet] = offsets[2] / 180.0 * D2PI / 2.0;
    printf("Angle detector %ld = %10.15g\n", idet, psi[idet]);
  }

  if (rank == 0)
    printf("nsegtot = %ld\n", nsegtot);

  // long *segnum;
  unsigned char *segon;
  long lnr = 30000;

  isampmin = new long[nsegtot];
  isampmax = new long[nsegtot];
  // segnum = new long[ndet * nsegtot];
  segon = new unsigned char[ndet * lnr];

  // for (ii = 0; ii < ndet * nsegtot; ii++)
  //   segnum[ii] = -1;

  for (ii = 0; ii < ndet * lnr; ii++)
    segon[ii] = 1; //// by default, all segments are used

  printf("LINE = %d \n", __LINE__);
  MPI_Barrier(MPI_COMM_WORLD);

  long *tmpsegnum;
  tmpsegnum = new long[2 * nsegtot];

  printf("Reading sample list file %s\n", filesegnum.c_str());

  #ifdef _DEBUG1
  printf("number of elements read = %ld\n", 2*nsegtot-1-0+1);
  printf("size of tmpsegnum array = %ld\n", 2*nsegtot);
  #endif
  for (idet = 0; idet < ndet; idet++) {
    //// READ LIST OF START SAMPLE FOR EACH SEGMENT
    MyErr = ReadLongVECT(tmpsegnum, filesegnum, 0, 2 * nsegtot - 1);
    printf("file_read status = %ld\n", MyErr);
    for (ii = 0; ii < nsegtot; ii++) {
      isampmin[ii] = tmpsegnum[2 * ii];
      isampmax[ii] = tmpsegnum[2 * ii + 1];
    }
  }

  printf("LINE = %d, ns = %ld, nssp = %ld \n", __LINE__, ns, nssp);
  MPI_Barrier(MPI_COMM_WORLD);

  //******************** Input data noise auto- and cross-power sp
  /// nssp = 1000000; ///// TO FIX!!
  string bolo;

  SpN = new double[nssp];

  if (CORRon) {
    printf("NOT IMPLEMENTED YET\n");
    exit(0);
  } else {
    SpN_all = new double[ndet * ndet * ns];
    for (ii = 0; ii < ns * ndet * ndet; ii++)
      SpN_all[ii] = 0.0;

    for (idet = 0; idet < ndet; idet++) {
      bolo = bolonames[idet];

      sprintf(NoiseSpFiledet, "%s%s", noiseSppreffile.c_str(), bolo.c_str());
      printf("NoiseSpFiledet = %s\n", NoiseSpFiledet);

      MyErr = ReadDoubleVECT(SpN, NoiseSpFiledet, 0, nssp - 1);
      if (rank == 0)
        printf("ns = %ld, %d\n", ns, MyErr);

      // printf("SpN[0] = %10.15g, SpN[1000] = %10.15g\n",SpN[0],SpN[1000]);

      for (ii = 0; ii < ns; ii++)
        SpN_all[ii + idet * ns + idet * ndet * ns] =
            SpN[long(double(ii) / nssp * ns)] / double(ns) / double(ns);
      SpN_all[idet * ns + idet * ndet * ns] =
          SpN[1] * 1e-6 / double(ns) / double(ns);
    }
  }

  //*********************** Optional Polar and Leakage parameters
  // if (0 & polar)
  // MyErr = PIOReadVECTObject((void **)&polardet, Param->Polardet, (char *)
  // "PIODOUBLE",  (const char *) "", NULL);
  polardet = new double[ndet];
  for (idet = 0; idet < ndet; idet++)
    polardet[idet] = 1.0;
  polarang = new double[ndet];
  for (idet = 0; idet < ndet; idet++)
    polarang[idet] = 0;
  relfGal = new double[ndet];
  for (idet = 0; idet < ndet; idet++)
    relfGal[idet] = 0;
  if (GalBP)
    MyErr = ReadVECT((void **)&relfGal, RelfGal, 8, 0, ndet - 1);

  /// polarang is an extra angle to apply to polarizer angles

  double ttmean = 0;
  double ttsig = 0;
  if (GalBP) {

    for (idet = 0; idet < ndet; idet++)
      ttmean += relfGal[idet] / double(ndet);
    for (idet = 0; idet < ndet; idet++)
      ttsig +=
          (relfGal[idet] - ttmean) * (relfGal[idet] - ttmean) / double(ndet);
    for (idet = 0; idet < ndet; idet++)
      relfGal[idet] = (relfGal[idet] - ttmean) / sqrt(ttsig);

    if (rank == 0)
      for (idet = 0; idet < ndet; idet++)
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! relfGal[%ld] = %10.15g\n", idet,
               relfGal[idet]);
  }

  if (rank == 0)
    if (CORRon)
      printf("CORRELATIONS BETWEEN DETECTORS INCLUDED\n");
  if (rank == 0)
    if (!CORRon)
      printf("NO CORRELATIONS BETWEEN DETECTORS INCLUDED\n");

  if (rank == 0)
    printf("%ld\n", nside);

  /// if parallelization with maps subsets
  // long nsubm = 4;  // for Planck analysis with nside 1024 or 2048
  long nsubm = 1; // for litebird with nside = 256

  /// MPI
  int rranks[1][3];
  rranks[0][0] = (rank / (size / nsubm)) * (size / nsubm);
  rranks[0][1] = (rank / (size / nsubm)) * (size / nsubm) + (size / nsubm) - 1;
  if (rranks[0][1] >= size)
    rranks[0][1] = size - 1;
  rranks[0][2] = 1;
  MPI_Group newgroup, mastergroup;
  MPI_Group MPICOMMGROUP;
  /// int ttt0 = MPI_Comm_group(MPI_COMM_WORLD, &MPICOMMGROUP);
  /// int ttt = MPI_Group_range_incl(MPICOMMGROUP,1,rranks,&newgroup);
  MPI_Comm_group(MPI_COMM_WORLD, &MPICOMMGROUP);
  MPI_Group_range_incl(MPICOMMGROUP, 1, rranks, &newgroup);

  if (nsubm > 1) {
    rranks[0][0] = 0;
    rranks[0][1] = ((size - 1) / (size / nsubm)) * (size / nsubm);
    rranks[0][2] = size / nsubm;
    /// ttt = MPI_Group_range_incl(MPICOMMGROUP,1,rranks,&mastergroup);
    MPI_Group_range_incl(MPICOMMGROUP, 1, rranks, &mastergroup);
  }

  // Create communicators
  MPI_Comm newcomm, mastercomm;
  MPI_Comm_create(MPI_COMM_WORLD, newgroup, &newcomm);
  if (nsubm > 1)
    MPI_Comm_create(MPI_COMM_WORLD, mastergroup, &mastercomm);

  int newrank;
  MPI_Comm_rank(newcomm, &newrank);
  MPI_Barrier(MPI_COMM_WORLD);

  //***** distribute data segments between procs
  if (newrank != 0) {
    iseg_min = ((rank % (size / nsubm)) - 1) * nsegtot / ((size / nsubm) - 1);
    iseg_max = (rank % (size / nsubm)) * nsegtot / ((size / nsubm) - 1) - 1;

    #ifdef _DEBUG1
    printf("newrank = %d, iseg_min = %ld, iseg_max = %ld\n", newrank, iseg_min, iseg_max);
    #endif

    nseg = iseg_max - iseg_min + 1;

    /********** Alocate memory ***********/
    ra = new double[ns * ndet * nseg];
    dec = new double[ns * ndet * nseg];
    datatopix = new long[ns * ndet * nseg];
    if (polar) {
      psip = new double[ns * ndet * nseg];
      if (WriteCrossLink != 0)
        psipFP = new double[ns * ndet * nseg];
    }
  }

  if (newrank != 0) {
    fdatas = new fftw_complex[ns * ndet];
    fdata = new fftw_complex[ns];
    fdatas0 = new fftw_complex[(ns / 2 + 1) * ndet * nseg];
    fodipole = new fftw_complex[(ns / 2 + 1) * ndet * nseg];
    odipole = new double[ns];
  }
  fftw_plan fftplan;

  printf("%s\n", bextension.c_str());

  printf("BEFORE READING DATA\n");

  ///*********************************** Read input data
  ///************************************//
  double *data, *orbdata, *ras, *decs, *psips;
  double ra_rad, dec_rad;

  if (newrank != 0) {

    /* read all input object only */

    for (iseg = iseg_min; iseg <= iseg_max; iseg++) {
      if (isampmax[iseg] - isampmin[iseg] + 1 != ns) {
        printf("Problem with segment size: %d, %d\n",
               isampmax[iseg] - isampmin[iseg], ns);
        exit(1);
      }

      ras = new double[ns];
      decs = new double[ns];
      if (polar)
        psips = new double[ns];

      // Read boresight pointing
      sprintf(namefilering, "%sra%s.bi", dirfilep.c_str(), pextension.c_str());
      printf("Reading pointing file: %s\n", namefilering);
      MyErr = ReadDoubleVECT(ras, namefilering, isampmin[iseg], isampmax[iseg]);

      sprintf(namefilering, "%sdec%s.bi", dirfilep.c_str(), pextension.c_str());
      MyErr =
          ReadDoubleVECT(decs, namefilering, isampmin[iseg], isampmax[iseg]);

      if (polar) {
        sprintf(namefilering, "%spsi%s.bi", dirfilep.c_str(),
                pextension.c_str());
        MyErr =
            ReadDoubleVECT(psips, namefilering, isampmin[iseg], isampmax[iseg]);
        // printf("psips[0] = %10.15g , psips[1895654] = %10.15g, psi[0] =
        // %10.15g, psi[1] = %10.15g, psi[2] = %10.15g, psi[3] =
        // %10.15g\n",psips[0],psips[1895654],idet,psi[0],psi[1],psi[2],psi[3]);
      }

      for (idet = 0; idet < ndet; idet++) {
        bolo = bolonames[idet];
        if (segon[idet * lnr + iseg]) {
          /* read orbital dipole data */
          if (recalib || odip) {
            orbdata = new double[ns];
            sprintf(namefilering, "%s/orbdip_%s", dirfile.c_str(),
                    bolo.c_str());
            MyErr = ReadDoubleVECT(orbdata, namefilering, isampmin[iseg],
                                   isampmax[iseg]);

            fftplan = fftw_plan_dft_r2c_1d(ns, orbdata, fdata, FFTW_ESTIMATE);
            fftw_execute(fftplan);
            fftw_destroy_plan(fftplan);
            for (ii = 0; ii < ns / 2 + 1; ii++) {
              fodipole[ii + idet * (ns / 2 + 1) +
                       (iseg - iseg_min) * ndet * (ns / 2 + 1)][0] =
                  fdata[ii][0];
              fodipole[ii + idet * (ns / 2 + 1) +
                       (iseg - iseg_min) * ndet * (ns / 2 + 1)][1] =
                  fdata[ii][1];
            }
          }

          for (ii = 0; ii < ns; ii++)
            if (!(psips[ii] > 0) && !(psips[ii] <= 0.0)) {
              printf("NAN DETECTED IN PSI, psi = %10.15g\n", psips[ii]);
              if (ii > 0)
                psips[ii] = psips[ii - 1];
            }

          /* Compute pointing for each detector */ //// need to add parameters
                                                   ///from the file
          double theta_o, phi_o, psi_o, thetap_i, phip_i, psip_i;
          for (ii = 0; ii < ns; ii++) {
            // xsh = offx[idet] * cos(psips[ii]) - offy[idet] * sin(psips[ii]);
            // ysh = offx[idet] * sin(psips[ii]) + offy[idet] * cos(psips[ii]);
            // slaDtp2s (xsh,ysh, ras[ii], decs[ii],&ra_rad,&dec_rad);

            thetap_i = D2PI / 4.0 - decs[ii];
            phip_i = ras[ii];
            psip_i = psips[ii];
            // pointingshift(-offx[idet],-offy[idet],1, &thetap_i, &phip_i,
            // &psip_i, &theta_o, &phi_o, &psi_o);  // To match with slaDtp2s()
            pointingshift(offx[idet], offy[idet], 1, &thetap_i, &phip_i,
                          &psip_i, &theta_o, &phi_o,
                          &psi_o); // Offset on the sky I am guessing

            ra_rad = phi_o;
            dec_rad = D2PI / 4.0 - theta_o;

            ra[ii + ns * (iseg - iseg_min) + ns * nseg * idet] =
                ra_rad; //// assuming radians
            dec[ii + ns * (iseg - iseg_min) + ns * nseg * idet] =
                dec_rad; ///// assuming radians
            // ra[ii+ns*(iseg-iseg_min)+ns*nseg*idet] = ras[ii];   //// assuming
            // radians dec[ii+ns*(iseg-iseg_min)+ns*nseg*idet] = decs[ii]; /////
            // assuming radians

            if (polar) {
              psip[ii + ns * (iseg - iseg_min) + ns * nseg * idet] =
                  psi_o + psi[idet]; //// add angle
              if (WriteCrossLink)
                psipFP[ii + ns * (iseg - iseg_min) + ns * nseg * idet] =
                    psip[ii + ns * (iseg - iseg_min) + ns * nseg * idet];
              if (HWP)
                psip[ii + ns * (iseg - iseg_min) + ns * nseg * idet] =
                    -(psi_o + psi[idet]) + 2.0 * OmegaHWP * (ii + ns * iseg);

              if ((ii < 0) & (iseg == 0))
                printf("psip[%ld]=%10.15g, angle bolo=%10.15g\n",
                       ii + ns * (iseg - iseg_min) + ns * nseg * idet,
                       psip[ii + ns * (iseg - iseg_min) + ns * nseg * idet],
                       psi[idet]);
            }
          }

          /* read all input object only */
          data = new double[ns];
          sprintf(namefilering, "%s/data_%s%s.bi", dirfile.c_str(),
                  bolo.c_str(), bextension.c_str()); // input toi

          if (iseg == iseg_min)
            printf("%s\n", namefilering);
          MyErr = ReadDoubleVECT(data, namefilering, isampmin[iseg],
                                 isampmax[iseg]);
          for (ii = 0; ii < ns; ii++)
            if (!(data[ii] > 0) && !(data[ii] <= 0.0)) {
              printf("NAN DETECTED IN %s, data[%ld] = %10.15g\n", namefilering,
                     ii, data[ii]);
              if (ii > 0)
                data[ii] = data[ii - 1];
            }

          if ((rank == 1) & (iseg == iseg_min))
            printf("data[0] = %10.15g\n", data[0]);

          fftplan = fftw_plan_dft_r2c_1d(ns, data, fdata, FFTW_ESTIMATE);
          fftw_execute(fftplan);
          fftw_destroy_plan(fftplan);
          for (ii = 0; ii < ns / 2 + 1; ii++) {
            fdatas0[ii + idet * (ns / 2 + 1) +
                    (iseg - iseg_min) * ndet * (ns / 2 + 1)][0] = fdata[ii][0];
            fdatas0[ii + idet * (ns / 2 + 1) +
                    (iseg - iseg_min) * ndet * (ns / 2 + 1)][1] = fdata[ii][1];
          }

          if (recalib) //// or odip??
            delete (orbdata);
          delete (data);
        }
      }
      delete (ras);
      delete (decs);
      if (polar)
        delete (psips);
      printf("DATA READ det %s, fraction %d/%ld\n", namefilering, newrank,
             (size - nsubm) / nsubm);
    }
  }

  // long vmem,phymem;

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    printf("DATA SUCCESSFULLY READ\n");

  printf("All OK, LINE = %d\n", __LINE__);
  MPI_Barrier(MPI_COMM_WORLD);

  // pixon indicates pixels that are seen
  pixon = new long[12 * nside * nside];
  for (ii = 0; ii < 12 * nside * nside; ii++)
    pixon[ii] = 0;

  printf("All OK, LINE = %d\n", __LINE__);
  MPI_Barrier(MPI_COMM_WORLD);

  //**********************************************************************************
  // loop to get coordinates of pixels that are seen
  //**********************************************************************************

  #ifdef _DEBUG1
  if(newrank == 2) {
    for(int i=0; i<50; ++i) {
      printf("ra[%d] = %lf, dec[%d] = %lf\n", i, ra[i], i, dec[i]);
    }
  }
  #endif

  if (newrank != 0) {
    long ipr_;
    ipr = 0;
    for (idet = 0; idet < ndet; idet++)
      for (iseg = 0; iseg < nseg; iseg++)
        if (segon[idet * lnr + iseg + iseg_min])
          for (ii = 0; ii < ns; ii++) {
            theta = D2PI / 4.0 - dec[ii + ns * iseg + ns * nseg * idet];
            // if (theta > D2PI/2.0)
            //   theta = D2PI/2.0 - theta;
            phi = fmod(ra[ii + ns * iseg + ns * nseg * idet] + D2PI, D2PI);
            ipr_ = ipr;
            ang2pix_ring(nside, theta, phi, &ipr);
            // printf("%ld\n",ipr);
            if (ipr < 0 || ipr > 12 * nside * nside - 1) {
              printf("PROBLEM WITH PIXEL INDEX: ii = %ld, ipr=%ld, ra = "
                     "%10.15g, dec=%10.15g\n",
                     ii, ipr, dec[ii + ns * iseg + ns * nseg * idet],
                     ra[ii + ns * iseg + ns * nseg * idet]);
              ipr = ipr_;
              printf("PROBLEM WITH PIXEL INDEX: ii = %ld, ipr=%ld, ra = "
                     "%10.15g, dec=%10.15g\n",
                     ii - 1, ipr, dec[ii - 1 + ns * iseg + ns * nseg * idet],
                     ra[ii - 1 + ns * iseg + ns * nseg * idet]);
            }
            pixon[ipr] += 1;
            datatopix[ii + ns * iseg + ns * nseg * idet] = ipr;
          }
    delete[] ra;
    delete[] dec;
  }

  printf("All OK, LINE = %d\n", __LINE__);
  MPI_Barrier(MPI_COMM_WORLD);

  pixonall = new long[12 * nside * nside];
  for (ii = 0; ii < 12 * nside * nside; ii++)
    pixonall[ii] = 0;

  MPI_Reduce(pixon, pixonall, 12 * nside * nside, MPI_DOUBLE, MPI_SUM, 0,
             newcomm);

  for (ii = 0; ii < 12 * nside * nside; ii++)
    pixon[ii] = pixonall[ii];

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(pixon, 12 * nside * nside, MPI_DOUBLE, 0, newcomm);

  delete[] pixonall;

  printf("All OK, LINE = %d\n", __LINE__);
  MPI_Barrier(MPI_COMM_WORLD);

  //***********************************************************************************
  //************** init mapmaking variables *************//

  indpix = new long[12 * nside * nside];
  for (ii = 0; ii < 12 * nside * nside; ii++)
    indpix[ii] = -1;

  ll = 0;
  for (ii = 0; ii < 12 * nside * nside; ii++) {
    if (pixon[ii] != 0) {
      indpix[ii] = ll;
      ll++;
    }
  }
  npix = ll;

  printf("npix = %ld, rank = %d \n", npix, rank);

  delete[] pixon;

  npixtot = npix;
  if (polar)
    npixtot += 2 * npix;
  if (GalBP)
    npixtot += npix;

  ///// implement here submap parallelization
  long ipixmin = (rank / (size / nsubm)) * npix / nsubm; /// = 0 is nsubm == 1
  long ipixmax = (rank / (size / nsubm) + 1) * npix / nsubm - 1;
  long nsmpix = ipixmax - ipixmin + 1;
  long pcount = 0;
  for (ii = 0; ii < 12 * nside * nside; ii++)
    if (indpix[ii] != -1) {
      if ((pcount < ipixmin) || (pcount > ipixmax))
        indpix[ii] = -indpix[ii] - 2;
      pcount += 1;
    }
  if (pcount != npix) {
    printf("Problem with the number of filled pixels\n");
    exit(1);
  }
  long nsmpixtot = nsmpix;
  if (polar)
    nsmpixtot += 2 * nsmpix;
  if (GalBP)
    nsmpixtot += nsmpix;

  printf("TOTAL NUMBER OF DETECTORS: %d\n", (int)ndet);
  printf("TOTAL NUMBER OF DATA SEGMENTS: %d\n", (int)nseg);
  printf("NUMBER OF SAMPLE PER DATA SEGMENT: %d\n", (int)ns);
  printf("TOTAL NUMBER OF FILLED PIXELS IN ONE MAP: %ld, TOTAL NUMBER: %ld, "
         "FOR PROCESSOR %d: %ld, NB OF HEALPIX PIXEL: %ld \n",
         npix, npixtot, rank, nsmpixtot, 12 * nside * nside);

  double varN;
  //// Modify noise spectrum to white and offset if set
  if (Destrip /* || simul */) {
    for (jj = 0; jj < ndet; jj++)
      for (kk = 0; kk < ndet; kk++) {
        //	  if (simul) {
        //	      varN = Param->simulSigmaNoise[idet] *
        //Param->simulSigmaNoise[idet]; 	      if (varN == 0.) varN = 1.; 	  } else {
        varN = SpN_all[ns / 10 + jj * ns + kk * ns * ndet] / double(ns);
        //	  }
        for (ii = 0; ii < ns; ii++)
          if (kk == jj)
            SpN_all[ii + jj * ns + kk * ns * ndet] = varN;
          else
            SpN_all[ii + jj * ns + kk * ns * ndet] = 0.0;
      }
    for (jj = 0; jj < ndet; jj++)
      SpN_all[jj * ns + jj * ns * ndet] =
          SpN_all[jj * ns + jj * ns * ndet] * 1e-6;
    printf("CAUTION!!!!!!!!!!!!!!!!!! : White noise + destriping \n");
  } else
    printf("Using input noise \n");

  //************************************************************************//
  //************************************************************************//
  // Pre-processing of data
  //************************************************************************//
  //************************************************************************//

  int iter;
  double var0, var00, var00_, var_n, var_n_, delta0, delta_n, delta_n_, delta_o,
      rtq, rtqt, alpha, beta, chi2, chi20, chi2tot, absrecal, absrecal_old;
  double *S, *tmpmap, *PtNPmatStot, *d, *Mp, *Mptot, *sumCross, *sumCrosstot,
      *qtot, *relCalib, *relCalib_old, *zeta, *gamma, *zeta_all, *gamma_all;
  long *hits, *hitstot, *tmptabl;
  double *tmptab, *tmpinitmap, *tmpinitmap2, *Nd;

  if (newrank != 0)
    Nd = new double[ns];

  relCalib = new double[ndet * nsegtot];
  relCalib_old = new double[ndet * nsegtot];
  zeta = new double[ndet * nsegtot];
  gamma = new double[ndet * nsegtot];
  zeta_all = new double[ndet];
  gamma_all = new double[ndet];
  initarray(relCalib, nsegtot * ndet, 1.0);
  zeros(gamma, nsegtot * ndet);
  zeros(zeta, nsegtot * ndet);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    printf("applycalib = %d \n", applycalib);

  if (applycalib) {
    double *tmpval = NULL;
    for (idet = 0; idet < ndet; idet++) {
      // read calib coeff from file
      // MyErr = ReadVECT((void **)&tmpval, Param->coefcalib[idet], 8, 0,
      // ndet-1);

      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
        printf("n_coef = %d, nRing = %ld \n", MyErr, nsegtot);

      for (iseg = 0; iseg < nsegtot; iseg++)
        relCalib[idet + iseg * ndet] = tmpval[iseg];
    }
    delete[] tmpval;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 2)
    for (idet = 0; idet < ndet; idet++)
      printf("RelCalib[%ld]=%10.15g\n", idet, relCalib[idet]);

  double *dipNd;
  dipNd = new double[nsegtot * ndet];
  zeros(dipNd, nsegtot * ndet);

  var0 = 1.0;
  var00 = 1.0;

  double StopCriterium = 1e-15;

  ////////////////////////////// STARTING ITERATIONS
  ////////////////////////////////////////////
  int rloop = 1;
  while (rloop && rloop < 500 && var0 / var00 > StopCriterium) {

    if ((recalib || applycalib) && (newrank != 0)) {
      for (iseg = 0; iseg < nseg; iseg++)
        for (idet = 0; idet < ndet; idet++)
          if (segon[idet * lnr + iseg + iseg_min])
            for (ii = 0; ii < ns / 2 + 1; ii++) {
              fdatas0[ii + idet * (ns / 2 + 1) + iseg * ndet * (ns / 2 + 1)]
                     [0] /= relCalib[idet + iseg + iseg_min * ndet];
              fdatas0[ii + idet * (ns / 2 + 1) + iseg * ndet * (ns / 2 + 1)]
                     [1] /= relCalib[idet + iseg + iseg_min * ndet];
            }
    }
    if ((recalib || odip) && (newrank != 0)) {
      for (idet = 0; idet < ndet; idet++)
        for (iseg = 0; iseg < nseg; iseg++)
          if (segon[idet * lnr + iseg + iseg_min])
            for (ii = 0; ii < ns / 2 + 1; ii++) {
              fdatas0[ii + idet * (ns / 2 + 1) + iseg * ndet * (ns / 2 + 1)]
                     [0] -= fodipole[ii + idet * (ns / 2 + 1) +
                                     iseg * ndet * (ns / 2 + 1)][0];
              fdatas0[ii + idet * (ns / 2 + 1) + iseg * ndet * (ns / 2 + 1)]
                     [1] -= fodipole[ii + idet * (ns / 2 + 1) +
                                     iseg * ndet * (ns / 2 + 1)][1];
            }
    }

    PNd = new double[nsmpixtot];
    PNdtot = new double[nsmpixtot];
    zeros(PNd, nsmpixtot);
    zeros(PNdtot, nsmpixtot);

    chi20 = 0.0;
    if (newrank != 0) {
      for (iseg = 0; iseg < nseg; iseg++) {

        for (idet = 0; idet < ndet; idet++)
          if (segon[idet * lnr + iseg + iseg_min])
            for (ii = 0; ii < ns / 2 + 1; ii++) {
              fdatas[ii + idet * ns][0] =
                  fdatas0[ii + idet * (ns / 2 + 1) + iseg * ndet * (ns / 2 + 1)]
                         [0];
              fdatas[ii + idet * ns][1] =
                  fdatas0[ii + idet * (ns / 2 + 1) + iseg * ndet * (ns / 2 + 1)]
                         [1];
            }

        for (idet = 0; idet < ndet; idet++) {
          // compute first term of chi2
          if (segon[idet * lnr + iseg + iseg_min]) {
            for (ii = 0; ii < ns / 2 + 1; ii++) {
              fdata[ii][0] = fdatas[ii + idet * ns][0] *
                             sqrt(SpN_all[ii + idet * ns + idet * ndet * ns]) *
                             relCalib[idet + (iseg + iseg_min) * ndet];
              fdata[ii][1] = fdatas[ii + idet * ns][1] *
                             sqrt(SpN_all[ii + idet * ns + idet * ndet * ns]) *
                             relCalib[idet + (iseg + iseg_min) * ndet];
            }
            // if ((rank == 1) & (iseg==0) & (idet==0)){
            //     FILE *fp=fopen("SpN.dat","w");
            //     fwrite(SpN_all+idet*ns+idet*ndet*ns, sizeof(double)*ns,
            //     1,fp); fclose(fp);
            // }
            for (ii = 0; ii < ns / 2 + 1; ii++)
              chi20 +=
                  (fdata[ii][0] * fdata[ii][0] + fdata[ii][1] * fdata[ii][1]) /
                  double(nsubm) * 2.0;
          }
        }

        do_PtNd(PNd, NULL, fdatas, datatopix, psip, SpN_all, ns, ndet, indpix,
                ipixmin, nsmpix, CORRon, iseg, iseg_min, nseg, segon, lnr,
                relCalib + (iseg + iseg_min) * ndet, NULL, NULL, polar,
                polardet, polarang, GalBP, relfGal, -1);

      } // end of iseg loop
    }

    MPI_Reduce(&chi20, &chi2tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&chi2tot, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    chi20 = chi2tot;

    MPI_Reduce(PNd, PNdtot, nsmpixtot, MPI_DOUBLE, MPI_SUM, 0, newcomm);
    MPI_Bcast(PNdtot, nsmpixtot, MPI_DOUBLE, 0, newcomm);

    //////////////////////////// force leakage to be 0 at high galactic latitude
    if (recalib && GalBP)
      forceLeakzero(PNdtot, nside, indpix, ipixmin, nsmpix, polar);

    if (rank == 0)
      printf("PtNd Computed...\n");

    printf("PNdtot[0] = %10.15g\n", PNdtot[0]);

    /////save PNd           /////////////////// maybe not OK for nsubm = 1
    if (0 & (newrank == 0)) {
      sprintf(mapoutstr, "%s_PNd%s", outdir.c_str(), termin_out.c_str());
      tmpinitmap2 = new double[npixtot];
      fMPI_merge(tmpinitmap2, PNdtot, nsmpix, npix, npixtot, rank, newrank,
                 size, nsubm, polar, GalBP, status, 0, mastercomm);
      if (rank == 0)
        write_reduced_map(mapoutstr, nside, tmpinitmap2, indpix);
      if (polar) {
        if (rank == 0) {
          sprintf(mapoutstr, "%s_Q_PNd%s", outdir.c_str(), termin_out.c_str());
          write_reduced_map(mapoutstr, nside, tmpinitmap2 + npix, indpix);
          sprintf(mapoutstr, "%s_U_PNd", outdir.c_str());
          write_reduced_map(mapoutstr, nside, tmpinitmap2 + 2 * npix, indpix);
        }
        if (GalBP && (rank == 0)) {
          sprintf(mapoutstr, "%s_Leak_PNd%s", outdir.c_str(),
                  termin_out.c_str());
          write_reduced_map(mapoutstr, nside, tmpinitmap2 + 3 * npix, indpix);
        }
      } else if (GalBP && (rank == 0)) {
        sprintf(mapoutstr, "%s_Leak_PNd%s", outdir.c_str(), termin_out.c_str());
        write_reduced_map(mapoutstr, nside, tmpinitmap2 + npix, indpix);
      }
      delete[] tmpinitmap2;
    }

    delete[] PNd;
    if (newrank != 0) {
      // delete[] datas;
      delete[] PNdtot;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //************************************************************************//
    //************************************************************************//
    // main loop here over data segments and detectors
    //************************************************************************//
    //************************************************************************//

    S = new double[nsmpixtot];
    if (rloop == 1) {
      Mp = new double[nsmpixtot];
      sumCross = new double[nsmpixtot];
    }
    tmpmap = new double[nsmpixtot];
    PtNPmatStot = new double[nsmpixtot];
    if (rloop == 1)
      hits = new long[nsmpix * 2];

    zeros(S, nsmpixtot);
    zeros(tmpmap, nsmpixtot);
    zeros(PtNPmatStot, nsmpixtot);
    if (rloop == 1) {
      zeros(Mp, nsmpixtot);
      zeros(sumCross, nsmpixtot);
    }
    if (rloop == 1)
      for (ii = 0; ii < nsmpix * 2; ii++)
        hits[ii] = 0;

    if (rloop == 1) {
      tmpinitmap = new double[npixtot];
      if (nsubm > 1)
        fMPI_merge(tmpinitmap, S, nsmpix, npix, npixtot, rank, newrank, size,
                   nsubm, polar, GalBP, status, 1, MPI_COMM_WORLD);
      else {
        // copy S to tmpinitmap
        for (ii = 0; ii < npixtot; ii++)
          tmpinitmap[ii] = S[ii];
      }

      if (initMapOn) {
        /////start iterations with a pre-computed map
        string filn;
        ////filn = "/wrk/akarakci/tmpinitmap.dat";
        // MyErr=ReadVECT((void **)&tmpinitmap, initMap, 8, 0, npixtot);
        MyErr = ReadDoubleVECT(tmpinitmap, initMap, 0, npixtot - 1);

        for (ii = 0; ii < nsmpix; ii++) {
          S[ii] = tmpinitmap[ii + ipixmin];
          if (GalBP)
            S[ii + nsmpix] = tmpinitmap[ii + ipixmin + npix];
          if (polar) {
            S[ii + nsmpix] = tmpinitmap[ii + ipixmin + npix];
            S[ii + 2 * nsmpix] = tmpinitmap[ii + ipixmin + 2 * npix];
            if (GalBP)
              S[ii + 3 * nsmpix] = tmpinitmap[ii + ipixmin + 3 * npix];
          }
        }
      }

    } else {
      for (ii = 0; ii < nsmpix; ii++) {
        S[ii] = tmpinitmap[ii + ipixmin];
        if (GalBP)
          S[ii + nsmpix] = tmpinitmap[ii + ipixmin + npix];
        if (polar) {
          S[ii + nsmpix] = tmpinitmap[ii + ipixmin + npix];
          S[ii + 2 * nsmpix] = tmpinitmap[ii + ipixmin + 2 * npix];
          if (GalBP)
            S[ii + 3 * nsmpix] = tmpinitmap[ii + ipixmin + 3 * npix];
        }
      }
    }

    //////////////////////////// force leakage to be 0 at high galactic latitude
    if (recalib && GalBP)
      forceLeakzero(S, nside, indpix, ipixmin, nsmpix,
                    polar); /// why tmpinitmap is not forced to 0??

    if (newrank != 0) {
      // S is erased to save memory
      for (iseg = 0; iseg < nseg; iseg++) {

        write_tfAS(tmpinitmap, fdatas, datatopix, indpix, psip, npix, ns, ndet,
                   iseg, iseg_min, nseg, segon, lnr, polar, polardet, polarang,
                   GalBP, relfGal, -1);

        if (rloop == 1) {
          if (WriteCrossLink) //// tmpmap is not calculated here
            do_PtNd(tmpmap, sumCross, fdatas, datatopix, psipFP, SpN_all, ns,
                    ndet, indpix, ipixmin, nsmpix, CORRon, iseg, iseg_min, nseg,
                    segon, lnr, relCalib + (iseg + iseg_min) * ndet, Mp, hits,
                    polar, polardet, polarang, GalBP, relfGal, -1);

          do_PtNd(tmpmap, NULL, fdatas, datatopix, psip, SpN_all, ns, ndet,
                  indpix, ipixmin, nsmpix, CORRon, iseg, iseg_min, nseg, segon,
                  lnr, relCalib + (iseg + iseg_min) * ndet, Mp, hits, polar,
                  polardet, polarang, GalBP, relfGal, -1);
        } else
          do_PtNd(tmpmap, NULL, fdatas, datatopix, psip, SpN_all, ns, ndet,
                  indpix, ipixmin, nsmpix, CORRon, iseg, iseg_min, nseg, segon,
                  lnr, relCalib + (iseg + iseg_min) * ndet, NULL, NULL, polar,
                  polardet, polarang, GalBP, relfGal, -1);

        // crosspar(mapscossin, datatopix, psip, ns, ndet, indpix, ipix_min,
        // npix,
        //    iseg, iseg_min, nseg, segon, lnr, polar, polardet, polarang, -1);

      } // end of iseg loop
    }
    if (newrank != 0)
      delete[] S;
    delete[] tmpinitmap;

    MPI_Reduce(tmpmap, PtNPmatStot, nsmpixtot, MPI_DOUBLE, MPI_SUM, 0, newcomm);
    if (newrank != 0)
      delete[] PtNPmatStot;

    long ntmp = nsmpix;
    if (polar)
      ntmp = 2 * nsmpix;

    if (rloop == 1) {
      hitstot = new long[ntmp];
      for (ii = 0; ii < ntmp; ii++)
        hitstot[ii] = 0;

      MPI_Reduce(hits, hitstot, ntmp, MPI_LONG, MPI_SUM, 0, newcomm);
      if (newrank != 0)
        delete[] hitstot;
      delete[] hits;

      Mptot = new double[nsmpixtot];
      sumCrosstot = new double[nsmpixtot];
      zeros(Mptot, nsmpixtot);
      zeros(sumCrosstot, nsmpixtot);
      MPI_Reduce(Mp, Mptot, nsmpixtot, MPI_DOUBLE, MPI_SUM, 0, newcomm);
      MPI_Reduce(sumCross, sumCrosstot, nsmpixtot, MPI_DOUBLE, MPI_SUM, 0,
                 newcomm);
      if (newrank != 0) {
        delete[] Mptot;
        delete[] sumCrosstot;
      }
      delete[] Mp;
      delete[] sumCross;
    }

    // remove some degeneracies
    if (newrank == 0)
      for (ii = 0; ii < nsmpixtot; ii++)
        if (ii >= nsmpix) {
          if (polar && (ii < 3 * nsmpix) &&
              (hitstot[(ii % nsmpix) + nsmpix] < 4)) {
            PtNPmatStot[ii] = 0.0;
            S[ii] = 0.0;
            PNdtot[ii] = 0.0;
            if (rloop == 1) {
              Mptot[ii] = 0.0;
              sumCrosstot[ii] = 0.0;
            }
          }
        }

    //////////////////////////// force leakage to be 0 at high galactic latitude
    if (recalib && GalBP && (newrank == 0))
      forceLeakzero(PtNPmatStot, nside, indpix, ipixmin, nsmpix, polar);

    d = new double[nsmpixtot];

    MPI_Barrier(MPI_COMM_WORLD);

    if (newrank == 0) {

      for (ii = 0; ii < nsmpixtot; ii++) {
        // Mptot[ii] = 1.0/double(hitstot[ii]);
        if ((Mptot[ii] == 0) && (ii < nsmpix)) {
          printf("ERROR: Mp[%ld] has elements = 0, ipixmin = %ld\n", ii,
                 ipixmin);
          exit(0);
        }
      }

      for (ii = 0; ii < nsmpixtot; ii++) {
        if ((rloop == 1) & (Mptot[ii] != 0.0))
          Mptot[ii] = 1.0 / Mptot[ii];
        PtNPmatStot[ii] = PNdtot[ii] - PtNPmatStot[ii];
        d[ii] = Mptot[ii] * PtNPmatStot[ii];
      }

      delta_n = 0.0;
      for (ii = 0; ii < nsmpixtot; ii++)
        delta_n += PtNPmatStot[ii] * d[ii];

      var_n = 0.0;
      for (ii = 0; ii < nsmpixtot; ii++)
        var_n += PtNPmatStot[ii] * PtNPmatStot[ii];

      var0 = 0.0;
      delta0 = 0.0;

      if (nsubm > 1) {
        MPI_Reduce(&var_n, &var0, 1, MPI_DOUBLE, MPI_SUM, 0, mastercomm);
        MPI_Reduce(&delta_n, &delta0, 1, MPI_DOUBLE, MPI_SUM, 0, mastercomm);
      } else {
        var0 = var_n;
        delta0 = delta_n;
      }

      if (rloop == 1) {
        var00_ = 0.0;
        for (ii = 0; ii < nsmpixtot; ii++)
          var00_ += PNdtot[ii] * PNdtot[ii];
        if (nsubm > 1)
          MPI_Reduce(&var00_, &var00, 1, MPI_DOUBLE, MPI_SUM, 0, mastercomm);
        else
          var00 = var00_;
      }

      if (nsubm > 1) { ///// CHECK IN CASE IT'S BROKEN
        MPI_Barrier(mastercomm);
        MPI_Bcast(&var0, 1, MPI_DOUBLE, 0, mastercomm);
        MPI_Bcast(&delta0, 1, MPI_DOUBLE, 0, mastercomm);
        if (rloop == 1)
          MPI_Bcast(&var00, 1, MPI_DOUBLE, 0, mastercomm);
      }

      // delta_n = delta0;
      // var_n = var0;
      if (rank == 0) {
        printf("var0 = %lg\n", var0);
        printf("var00 = %lg\n", var00);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(d, nsmpixtot, MPI_DOUBLE, 0, newcomm);
    MPI_Bcast(&var0, 1, MPI_DOUBLE, 0, newcomm);
    MPI_Bcast(&var_n, 1, MPI_DOUBLE, 0, newcomm);
    MPI_Bcast(&var00, 1, MPI_DOUBLE, 0, newcomm);

    if (newrank == 0)
      qtot = new double[nsmpixtot];

    ////////////////////////////////start loop
    ////////////////////////////////////////////////////
    iter = 0;
    while ((iter < 2000 && var_n / var00 > StopCriterium) || iter == 0) {

      zeros(tmpmap, nsmpixtot);

      //// distribute d to a full map
      tmpinitmap = new double[npixtot];
      if (nsubm > 1)
        fMPI_merge(tmpinitmap, d, nsmpix, npix, npixtot, rank, newrank, size,
                   nsubm, polar, GalBP, status, 1, MPI_COMM_WORLD);
      else
        for (ii = 0; ii < npixtot; ii++)
          tmpinitmap[ii] = d[ii];

      if (newrank != 0) {
        for (iseg = 0; iseg < nseg; iseg++) {

          write_tfAS(tmpinitmap, fdatas, datatopix, indpix, psip, npix, ns,
                     ndet, iseg, iseg_min, nseg, segon, lnr, polar, polardet,
                     polarang, GalBP, relfGal, -1);

          do_PtNd(tmpmap, NULL, fdatas, datatopix, psip, SpN_all, ns, ndet,
                  indpix, ipixmin, nsmpix, CORRon, iseg, iseg_min, nseg, segon,
                  lnr, relCalib + (iseg + iseg_min) * ndet, NULL, NULL, polar,
                  polardet, polarang, GalBP, relfGal, -1);
        } // end of iseg loop

        delete[] d;
        qtot = new double[nsmpixtot];
      }

      delete[] tmpinitmap;

      zeros(qtot, nsmpixtot);
      MPI_Reduce(tmpmap, qtot, nsmpixtot, MPI_DOUBLE, MPI_SUM, 0, newcomm);
      if (newrank != 0)
        delete[] qtot;

      if (newrank == 0) {
        // remove degeneracies by forcing to 0 the Q and U components for which
        // the nb of obs is < 4
        for (ii = 0; ii < nsmpixtot; ii++)
          if (ii >= nsmpix)
            if (polar && (ii < 3 * nsmpix) &&
                (hitstot[(ii % nsmpix) + nsmpix] < 4))
              qtot[ii] = 0.0;

        //////////////////////////// force leakage to be 0 at high galactic
        ///latitude
        if (recalib && GalBP)
          forceLeakzero(qtot, nside, indpix, ipixmin, nsmpix, polar);

        rtq = 0.0;
        for (ii = 0; ii < nsmpixtot; ii++)
          rtq += qtot[ii] * d[ii];
        rtqt = 0.0;

        if (nsubm > 1) {
          MPI_Reduce(&rtq, &rtqt, 1, MPI_DOUBLE, MPI_SUM, 0, mastercomm);
          MPI_Bcast(&rtqt, 1, MPI_DOUBLE, 0, mastercomm);
        } else
          rtqt = rtq;

        alpha = delta_n / rtqt;

        for (ii = 0; ii < nsmpixtot; ii++)
          S[ii] += alpha * d[ii];
      }

      if ((iter % 10) == 0) {

        //// distribute S to a full map
        tmpinitmap = new double[npixtot];

        if (nsubm > 1)
          fMPI_merge(tmpinitmap, S, nsmpix, npix, npixtot, rank, newrank, size,
                     nsubm, polar, GalBP, status, 1, MPI_COMM_WORLD);
        else {
          if (newrank == 0)
            for (ii = 0; ii < npixtot; ii++)
              tmpinitmap[ii] = S[ii];
          MPI_Bcast(tmpinitmap, npixtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        zeros(tmpmap, nsmpixtot);

        chi2 = 0.0;
        chi2tot = 0.0;
        if (newrank != 0) {
          for (iseg = 0; iseg < nseg; iseg++) {

            write_tfAS(tmpinitmap, fdatas, datatopix, indpix, psip, npix, ns,
                       ndet, iseg, iseg_min, nseg, segon, lnr, polar, polardet,
                       polarang, GalBP, relfGal, -1);

            do_PtNd(tmpmap, NULL, fdatas, datatopix, psip, SpN_all, ns, ndet,
                    indpix, ipixmin, nsmpix, CORRon, iseg, iseg_min, nseg,
                    segon, lnr, relCalib + (iseg + iseg_min) * ndet, NULL, NULL,
                    polar, polardet, polarang, GalBP, relfGal, -1);

            //// compute chi2
            for (idet = 0; idet < ndet; idet++) {
              if (segon[idet * lnr + iseg + iseg_min]) {
                // compute first term of chi2
                for (ii = 0; ii < ns / 2 + 1; ii++) {
                  fdata[ii][0] =
                      (fdatas0[ii + idet * (ns / 2 + 1) +
                               iseg * ndet * (ns / 2 + 1)][0] -
                       fdatas[ii + idet * ns][0]) *
                      sqrt(SpN_all[ii + idet * ns + idet * ndet * ns]) *
                      relCalib[idet + (iseg + iseg_min) * ndet];
                  fdata[ii][1] =
                      (fdatas0[ii + idet * (ns / 2 + 1) +
                               iseg * ndet * (ns / 2 + 1)][1] -
                       fdatas[ii + idet * ns][1]) *
                      sqrt(SpN_all[ii + idet * ns + idet * ndet * ns]) *
                      relCalib[idet + (iseg + iseg_min) * ndet];
                }

                for (ii = 0; ii < ns / 2 + 1; ii++)
                  chi2 += (fdata[ii][0] * fdata[ii][0] +
                           fdata[ii][1] * fdata[ii][1]) /
                          double(nsubm) * 2.0;
              }
            }
          } // end of iseg loop
        }
        delete[] tmpinitmap;

        // Chi2:
        MPI_Reduce(&chi2, &chi2tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Bcast(&chi2tot, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        chi2 = chi2tot;

        if (newrank != 0)
          PtNPmatStot = new double[nsmpixtot];
        zeros(PtNPmatStot, nsmpixtot);

        MPI_Reduce(tmpmap, PtNPmatStot, nsmpixtot, MPI_DOUBLE, MPI_SUM, 0,
                   newcomm);

        if (newrank != 0)
          delete[] PtNPmatStot;
        if (newrank == 0) {
          // remove degeneracies
          for (ii = 0; ii < nsmpixtot; ii++)
            if (ii >= nsmpix)
              if (polar && (ii < 3 * nsmpix) &&
                  (hitstot[(ii % nsmpix) + nsmpix] < 4))
                PtNPmatStot[ii] = 0.0;
          for (ii = 0; ii < nsmpixtot; ii++)
            PtNPmatStot[ii] = PNdtot[ii] - PtNPmatStot[ii];
        }

        //////////////////////////// force leakage to be 0 at high galactic
        ///latitude
        if (recalib && GalBP && newrank == 0)
          forceLeakzero(PtNPmatStot, nside, indpix, ipixmin, nsmpix, polar);

      } else ////////////  In case iteration number is not a multiple of 10
        if (newrank == 0)
          for (ii = 0; ii < nsmpixtot; ii++)
            PtNPmatStot[ii] -= alpha * qtot[ii];

      if (newrank == 0) {
        delta_o = delta_n;
        delta_n_ = 0.0;
        for (ii = 0; ii < nsmpixtot; ii++)
          delta_n_ += PtNPmatStot[ii] * Mptot[ii] * PtNPmatStot[ii];

        var_n_ = 0.0;
        for (ii = 0; ii < nsmpixtot; ii++)
          var_n_ += PtNPmatStot[ii] * PtNPmatStot[ii];

        delta_n = 0.0;
        var_n = 0.0;

        if (nsubm > 1) {
          MPI_Reduce(&delta_n_, &delta_n, 1, MPI_DOUBLE, MPI_SUM, 0,
                     mastercomm);
          MPI_Reduce(&var_n_, &var_n, 1, MPI_DOUBLE, MPI_SUM, 0, mastercomm);
          MPI_Barrier(mastercomm);
          MPI_Bcast(&var_n, 1, MPI_DOUBLE, 0, mastercomm);
          MPI_Bcast(&delta_n, 1, MPI_DOUBLE, 0, mastercomm);
        } else {
          delta_n = delta_n_;
          var_n = var_n_;
        }

        beta = delta_n / delta_o;
        for (ii = 0; ii < nsmpixtot; ii++)
          d[ii] = Mptot[ii] * PtNPmatStot[ii] + beta * d[ii];

        if (rank == 0)
          printf("iter = %d, crit = %10.15g, crit2 = %10.15g, chi2 = %10.15g\n",
                 iter, var_n / var00, delta_n / delta0, chi2tot);
      }

      //************************************************************  Write
      //variance maps Interation 1
      //****************************************************************//

      if (iter == 1 && newrank == 0 && rloop == 1) {

        sprintf(mapoutstr, "%s_NVAR%s", outdir.c_str(), termin_out.c_str());
        tmpinitmap = new double[npixtot];
        if (nsubm > 1)
          fMPI_merge(tmpinitmap, Mptot, nsmpix, npix, npixtot, rank, newrank,
                     size, nsubm, polar, GalBP, status, 0, mastercomm);
        else {
          for (ii = 0; ii < npixtot; ii++)
            tmpinitmap[ii] = Mptot[ii];
        }
        if (rank == 0)
          write_reduced_map(mapoutstr, nside, tmpinitmap, indpix);
        sprintf(mapoutstr, "%s_HITS%s", outdir.c_str(), termin_out.c_str());
        tmptabl = new long[npix];
        tmptab = new double[npix];
        if (nsubm > 1) {
          if (rank != 0)
            MPI_Send(hitstot, nsmpix, MPI_LONG, 0, rank, MPI_COMM_WORLD);
          if (rank == 0) {
            for (ii = 0; ii < nsmpix; ii++)
              tmptabl[ii] = hitstot[ii];
            for (idet = 1; idet < nsubm; idet++)
              MPI_Recv(tmptabl + idet * npix / nsubm,
                       -idet * npix / nsubm + (idet + 1) * npix / nsubm,
                       MPI_LONG, idet * size / nsubm, idet * size / nsubm,
                       MPI_COMM_WORLD, &status);
            for (ii = 0; ii < npix; ii++)
              tmptab[ii] = double(tmptabl[ii]);
            write_reduced_map(mapoutstr, nside, tmptab, indpix);
          }
        } else {
          /// just copy data
          for (ii = 0; ii < npix; ii++)
            tmptab[ii] = double(hitstot[ii]);
          write_reduced_map(mapoutstr, nside, tmptab, indpix);
          sprintf(mapoutstr, "%s_COS2P_%s", outdir.c_str(), termin_out.c_str());
          write_reduced_map(mapoutstr, nside, sumCrosstot, indpix);
          sprintf(mapoutstr, "%s_SIN2P_%s", outdir.c_str(), termin_out.c_str());
          write_reduced_map(mapoutstr, nside, sumCrosstot + npix, indpix);
        }

        if (polar) {

          if (rank == 0) {
            sprintf(mapoutstr, "%s_Q_NVAR%s", outdir.c_str(),
                    termin_out.c_str());
            write_reduced_map(mapoutstr, nside, tmpinitmap + npix, indpix);
            sprintf(mapoutstr, "%s_U_NVAR%s", outdir.c_str(),
                    termin_out.c_str());
            write_reduced_map(mapoutstr, nside, tmpinitmap + 2 * npix, indpix);
          }
          if (nsubm > 1) {
            if (rank != 0)
              MPI_Send(hitstot + nsmpix, nsmpix, MPI_LONG, 0, rank,
                       MPI_COMM_WORLD);
            if (rank == 0) {
              for (ii = 0; ii < nsmpix; ii++)
                tmptabl[ii] = hitstot[ii + nsmpix];
              for (idet = 1; idet < nsubm; idet++)
                MPI_Recv(tmptabl + idet * npix / nsubm,
                         -idet * npix / nsubm + (idet + 1) * npix / nsubm,
                         MPI_LONG, idet * size / nsubm, idet * size / nsubm,
                         MPI_COMM_WORLD, &status);
              for (ii = 0; ii < npix; ii++)
                tmptab[ii] = double(tmptabl[ii]);
              sprintf(mapoutstr, "%s_POL_HITS%s", outdir.c_str(),
                      termin_out.c_str());
              write_reduced_map(mapoutstr, nside, tmptab, indpix);
            }
          } else {
            /// just copy data
            for (ii = 0; ii < npix; ii++)
              tmptab[ii] = double(hitstot[ii + npix]);
            sprintf(mapoutstr, "%s_POL_HITS%s", outdir.c_str(),
                    termin_out.c_str());
            write_reduced_map(mapoutstr, nside, tmptab, indpix);
          }
          if (GalBP && (rank == 0)) {
            sprintf(mapoutstr, "%s_Leak_NVAR%s", outdir.c_str(),
                    termin_out.c_str());
            write_reduced_map(mapoutstr, nside, tmpinitmap + 3 * npix, indpix);
          }
        } else if (GalBP && (rank == 0)) {
          sprintf(mapoutstr, "%s_Leak_NVAR%s", outdir.c_str(),
                  termin_out.c_str());
          write_reduced_map(mapoutstr, nside, tmpinitmap + npix, indpix);
        }
        delete[] tmptab;
        delete[] tmptabl;
        delete[] tmpinitmap;
      } //******************************************************* End of
        //variance writting at iteration 1
        //**************************************************************//

      if ((newrank == 0) && iterw && (iter % iterw) == 0 && (rloop == 1)) {

        sprintf(mapoutstr, "%s%d", outdir.c_str(), iter);
        tmpinitmap = new double[npixtot];
        if (nsubm > 1)
          fMPI_merge(tmpinitmap, S, nsmpix, npix, npixtot, rank, newrank, size,
                     nsubm, polar, GalBP, status, 0, mastercomm);
        else
          for (ii = 0; ii < npixtot; ii++)
            tmpinitmap[ii] = S[ii];

        if (writeIterOn) {
          if (rank == 0) {
            sprintf(mapoutstr, "%s_I%s_%d", outdir.c_str(), termin_out.c_str(),
                    iter);
            write_reduced_map(mapoutstr, nside, tmpinitmap, indpix);
          }

          if (polar && (rank == 0)) {

            sprintf(mapoutstr, "%s_Q%s_%d", outdir.c_str(), termin_out.c_str(),
                    iter);
            write_reduced_map(mapoutstr, nside, tmpinitmap + npix, indpix);
            sprintf(mapoutstr, "%s_U%s_%d", outdir.c_str(), termin_out.c_str(),
                    iter);
            write_reduced_map(mapoutstr, nside, tmpinitmap + 2 * npix, indpix);

            if (GalBP && (rank == 0)) {
              sprintf(mapoutstr, "%s_Leak%s_%d", outdir.c_str(),
                      termin_out.c_str(), iter);
              write_reduced_map(mapoutstr, nside, tmpinitmap + 3 * npix,
                                indpix);
            }
          } else if (GalBP && (rank == 0)) {
            sprintf(mapoutstr, "%s_Leak%s_%d", outdir.c_str(),
                    termin_out.c_str(), iter);
            write_reduced_map(mapoutstr, nside, tmpinitmap + npix, indpix);
          }
        }
        delete[] tmpinitmap;
      }

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Bcast(&var_n, 1, MPI_DOUBLE, 0, newcomm);

      if (newrank != 0)
        d = new double[nsmpixtot];
      MPI_Bcast(d, nsmpixtot, MPI_DOUBLE, 0, newcomm);

      iter++;
    } //////////////// Map Iteration loop

    if (newrank == 0)
      printf("\n");

    // printf("OK0 rank = %ld\n",rank);

    MPI_Barrier(MPI_COMM_WORLD);

    //******************************  write final maps in file
    //********************************

    if (newrank == 0) {

      tmpinitmap = new double[npixtot];
      if (nsubm > 1)
        fMPI_merge(tmpinitmap, S, nsmpix, npix, npixtot, rank, newrank, size,
                   nsubm, polar, GalBP, status, 0, mastercomm);
      else {
        for (ii = 0; ii < npixtot; ii++)
          tmpinitmap[ii] = S[ii];
      }

      if (rank == 0) {
        //// Write map globaly
        sprintf(mapoutstr, "%s_A_%s_%d", outdir.c_str(), termin_out.c_str(),
                rloop);
        NbWrite = WriteMAP(tmpinitmap, mapoutstr, 8, npixtot);

        sprintf(mapoutstr, "%s_I_%s_%d", outdir.c_str(), termin_out.c_str(),
                rloop);
        write_reduced_map(mapoutstr, nside, tmpinitmap, indpix);

        if (polar) {

          sprintf(mapoutstr, "%s_Q_%s_%d", outdir.c_str(), termin_out.c_str(),
                  rloop);
          write_reduced_map(mapoutstr, nside, tmpinitmap + npix, indpix);
          sprintf(mapoutstr, "%s_U_%s_%d", outdir.c_str(), termin_out.c_str(),
                  rloop);
          write_reduced_map(mapoutstr, nside, tmpinitmap + 2 * npix, indpix);

          if (GalBP) {
            sprintf(mapoutstr, "%s_Leak_%s_%d", outdir.c_str(),
                    termin_out.c_str(), rloop);
            write_reduced_map(mapoutstr, nside, tmpinitmap + 3 * npix, indpix);
          }
        } else if (GalBP) {
          sprintf(mapoutstr, "%s_Leak_%s_%d", outdir.c_str(),
                  termin_out.c_str(), rloop);
          write_reduced_map(mapoutstr, nside, tmpinitmap + npix, indpix);
        }
      }
      delete[] S;
      if (!recalLeak && !recalPolar && !recalib)
        delete[] tmpinitmap;
    }

    //************************** End of map-making loop
    //*********************************//
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
      printf("Map-making Ends \n");

    //*****************************  If recalibrate leakage component
    //*************************//

    if (((GalBP && recalLeak) || (polar && recalPolar))) {
      // Recalib galactic leakage and polar
      // For this, need to compute r_{0,1,2,3} = N^(-1) Ps for s = I, GalBP, Q',
      // U' where Q' and U' are Q,U rotated to focalplane reference frame From
      // these 3 TOI-length data r_{1,2,3}, compute b_i = sum([m-r0].r_i) and
      // M_ij = sum(r_i.r_j) And finally solve for K = M^(-1).b

      double *r0 = NULL, *r1 = NULL, *r2 = NULL, *r3 = NULL;
      double *relfGal_n = NULL, *polardet_n = NULL, *polarang_n = NULL;

      if (newrank != 0)
        tmpinitmap = new double[npixtot];
      MPI_Bcast(tmpinitmap, npixtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      if (newrank != 0)
        r0 = new double[ns * nseg];

      if (GalBP) {
        if (newrank != 0)
          r1 = new double[ns * nseg];
        relfGal_n = new double[ndet];
      }
      if (polar) {
        if (newrank != 0) {
          r2 = new double[ns * nseg];
          r3 = new double[ns * nseg];
        }
        polardet_n = new double[ndet];
        polarang_n = new double[ndet];
      }

      for (idet = 0; idet < ndet;
           idet++) { // Calibration for each detector is independent
        if (newrank != 0) {
          for (iseg = iseg_min; iseg <= iseg_max; iseg++) {
            // void deproject_partial(double *S, long *indpix, long *datatopix,
            // long ns, double *Ps)
            if (segon[idet * lnr + iseg])
              deproject_partial(tmpinitmap, indpix,
                                datatopix + ns * (iseg - iseg_min) +
                                    ns * nseg * idet,
                                ns, r0 + ns * (iseg - iseg_min));
          }
        }

        if ((GalBP || recalLeak) && newrank != 0) {
          long pixshift;
          if (polar)
            pixshift = 3 * npix; // Leakage map = S + 3 * npix
          else
            pixshift = npix; // Leakage map = S + npix
          for (iseg = iseg_min; iseg <= iseg_max; iseg++) {
            if (segon[idet * lnr + iseg])
              deproject_partial(tmpinitmap + pixshift, indpix,
                                datatopix + ns * (iseg - iseg_min) +
                                    ns * nseg * idet,
                                ns, r1 + ns * (iseg - iseg_min));
          }
        }
        if ((polar || recalPolar) && newrank != 0) {
          long pixshift;
          pixshift = npix; // Q map = S + npix
          for (iseg = iseg_min; iseg <= iseg_max; iseg++) {
            if (segon[idet * lnr + iseg])
              deproject_partial(tmpinitmap + pixshift, indpix,
                                datatopix + ns * (iseg - iseg_min) +
                                    ns * nseg * idet,
                                ns, r2 + ns * (iseg - iseg_min));
          }
          pixshift = 2 * npix; // U map = S + 2 * npix
          for (iseg = iseg_min; iseg <= iseg_max; iseg++)
            if (segon[idet * lnr + iseg])
              deproject_partial(tmpinitmap + pixshift, indpix,
                                datatopix + ns * (iseg - iseg_min) +
                                    ns * nseg * idet,
                                ns, r3 + ns * (iseg - iseg_min));
          // Rotate Q and U to reference frame
          for (ii = 0; ii < nseg * ns; ii++) {
            double q, u;
            double qp, up, c, s;
            q = r2[ii];
            u = r3[ii];
            c = cos(2 * psip[ii + ns * nseg * idet]);
            s = sin(2 * psip[ii + ns * nseg * idet]);
            qp = q * c + u * s;
            up = -q * s + u * c;
            r2[ii] = qp;
            r3[ii] = up;
          }
        }

        // Build and solve the system for calibration coefficients
        if (recalLeak && !recalPolar) { // Simple case : scalar
          // polardet = Sum((datas[ii] - r0[ii])*r1[ii]) / Sum(r1[ii]*r1[ii])
          double A_, b_, A, b;
          A_ = 0.0;
          b_ = 0.0;
          if (newrank != 0) {
            double *Nm1r1;
            Nm1r1 = new double[ns];
            for (iseg = iseg_min; iseg <= iseg_max; iseg++) {
              if (segon[idet * lnr + iseg]) {
                compute_Nm1d(r1 + ns * (iseg - iseg_min), ns,
                             SpN_all + idet * ns + idet * ndet * ns,
                             recalib ? relCalib[idet + iseg * ndet] : 1.,
                             Nm1r1);
                for (ii = 0; ii < ns; ii++) {
                  long idx;
                  double Ps;
                  idx = (iseg - iseg_min) * ns + ii;
                  Ps = r0[idx];
                  if (polar)
                    Ps += polardet[idet] * (r2[idx] * cos(2 * polarang[idet]) +
                                            r3[idx] * sin(2 * polarang[idet]));
                  A_ += r1[idx] * Nm1r1[ii];
                  // b_ += (datas[idx + idet * ns * nseg] - Ps) * Nm1r1[ii];
                  // ////////  Implement fitting in Fourier space
                }
              }
            }
            delete[] Nm1r1;
          }
          // Reduce among processors
          MPI_Allreduce(&A_, &A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          MPI_Allreduce(&b_, &b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          // Divide by number of submap
          b /= nsubm;
          A /= nsubm;
          // All processor can compute the new leakage calib
          relfGal_n[idet] = b / A;
          // DEBUG
          if (newrank == 1)
            printf("rank=%d : idet = %d : A_ = %.10lg   b_ = %.10lg   A = "
                   "%.10lg   b = %.10lg\n",
                   rank, (int)idet, A_, b_, A, b);
        }
        if (!recalLeak &&
            recalPolar) { // Need to build a 2x2 matrix and solve the system
          /*
          double A_[4], b_[2], A[4], b[2], x[2];
          zeros(A_,4);
          zeros(b_,2);
          if (newrank != 0){
            double * Nm1r2, * Nm1r3;
            Nm1r2 = new double[ns];
            Nm1r3 = new double[ns];
            for (iseg = iseg_min; iseg <= iseg_max; iseg ++) {
              if (segon[idet*lnr + iseg]){
                compute_Nm1d (r2 + ns * (iseg - iseg_min), ns,
                              SpN_all + idet * ns + idet * ndet * ns,
                              recalib ? relCalib[idet + iseg * ndet] : 1.,
          Nm1r2); compute_Nm1d (r3 + ns * (iseg - iseg_min), ns, SpN_all + idet
          * ns + idet * ndet * ns, recalib ? relCalib[idet + iseg * ndet] : 1.,
          Nm1r3); for (ii = 0; ii < ns; ii ++){ long idx; double Ps; idx = (iseg
          - iseg_min) * ns + ii; Ps = r0[idx]; if (GalBP) Ps += r1[idx]; A_[0]
          += r2[idx] * Nm1r2[ii]; A_[1] += r2[idx] * Nm1r3[ii]; A_[3] += r3[idx]
          * Nm1r3[ii]; b_[0] += (datas[idx + idet * ns * nseg] - Ps) *
          Nm1r2[ii]; b_[1] += (datas[idx + idet * ns * nseg] - Ps) * Nm1r3[ii];
                }
              }
            }
            delete [] Nm1r2;
            delete [] Nm1r3;
          }
          A_[2] = A_[1]; // Symmetric matrix
          // Reduce among processors
          MPI_Allreduce(A_, A, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          MPI_Allreduce(b_, b, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          // All processors can solve the new polar calib
          NagError fail;
          SET_FAIL (fail);
          nag_real_lin_eqn(2, A, 2, b, x, &fail);
          polardet_n[idet] = sqrt (x[0] * x[0] + x[1] * x[1]);
          polarang_n[idet] = 0.5 * atan2 (x[1], x[0]);
          */
        }
        if (recalLeak &&
            recalPolar) { // Need to build a 3x3 matrix and solve the system
          /*
          double A_[9], b_[3], A[9], b[3], x[3];
          zeros(A_,9);
          zeros(b_,3);
          if (newrank != 0){
            double * Nm1r1, * Nm1r2, * Nm1r3;
            Nm1r1 = new double[ns];
            Nm1r2 = new double[ns];
            Nm1r3 = new double[ns];
            for (iseg = iseg_min; iseg <= iseg_max; iseg ++){
              if (segon[idet*lnr + iseg]){
                compute_Nm1d (r1 + ns * (iseg - iseg_min),
                              ns, SpN_all + idet * ns + idet * ndet * ns,
                              recalib ? relCalib[idet + iseg * ndet] : 1.,
          Nm1r1); compute_Nm1d (r2 + ns * (iseg - iseg_min), ns, SpN_all + idet
          * ns + idet * ndet * ns, recalib ? relCalib[idet + iseg * ndet] : 1.,
          Nm1r2); compute_Nm1d (r3 + ns * (iseg - iseg_min), ns, SpN_all + idet
          * ns + idet * ndet * ns, recalib ? relCalib[idet + iseg * ndet] : 1.,
          Nm1r3); for (ii = 0; ii < ns; ii ++){ long idx; idx = (iseg -
          iseg_min) * ns + ii; A_[0] += r1[idx] * Nm1r1[ii]; A_[1] += r1[idx] *
          Nm1r2[ii]; A_[2] += r1[idx] * Nm1r3[ii]; A_[4] += r2[idx] * Nm1r2[ii];
                  A_[5] += r2[idx] * Nm1r3[ii];
                  A_[8] += r3[idx] * Nm1r3[ii];
                  b_[0] += (datas[idx + idet * ns * nseg] - r0[idx]) *
          Nm1r1[ii]; b_[1] += (datas[idx + idet * ns * nseg] - r0[idx]) *
          Nm1r2[ii]; b_[2] += (datas[idx + idet * ns * nseg] - r0[idx]) *
          Nm1r3[ii];
                }
              }
            }
            delete [] Nm1r1;
            delete [] Nm1r2;
            delete [] Nm1r3;
          }
          A_[3] = A_[1]; // Symmetric matrix
          A_[6] = A_[2]; // Symmetric matrix
          A_[7] = A_[5]; // Symmetric matrix
          // Reduce among processors
          MPI_Allreduce(A_, A, 9, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          MPI_Allreduce(b_, b, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          // All processors can solve the new polar calib
          NagError fail;
          SET_FAIL (fail);
          nag_real_lin_eqn(3, A, 3, b, x, &fail);
          relfGal_n[idet] = x[0];
          polardet_n[idet] = sqrt (x[1] * x[1] + x[2] * x[2]);
          polarang_n[idet] = 0.5 * atan2 (x[2], x[1]);
          */
        }
      } // End of Loop on detectors

      // Apply normalization and fix degeneracies
      if (recalLeak) {
        double ttmean = 0;
        double ttsig = 0;
        for (idet = 0; idet < ndet; idet++)
          ttmean += relfGal_n[idet] / double(ndet);
        for (idet = 0; idet < ndet; idet++)
          ttsig += (relfGal_n[idet] - ttmean) * (relfGal_n[idet] - ttmean) /
                   double(ndet);
        for (idet = 0; idet < ndet; idet++)
          relfGal[idet] = (relfGal_n[idet] - ttmean) / sqrt(ttsig);

        if (rank == 0)
          for (idet = 0; idet < ndet; idet++)
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! relfGal[%ld] = %10.15g\n",
                   idet, relfGal[idet]);
      }

      if (recalPolar) {
        // Do not change angle and cross-polar of first detector
        for (idet = 1; idet < ndet; idet++) {
          polardet[idet] = polardet_n[idet];
          polarang[idet] = polarang_n[idet];
        }

        if (rank == 0)
          for (idet = 0; idet < ndet; idet++)
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! polardet[%ld] = %10.15g    "
                   "polarang[%ld] = %10.15g\n",
                   idet, polardet[idet], idet, polarang[idet] * 180 / M_PI);
      }

      // Free all allocated memory, except tmpinitmap
      if (r0)
        delete[] r0;
      if (relfGal_n)
        delete[] relfGal_n;
      if (r1)
        delete[] r1;
      if (r2)
        delete[] r2;
      if (r3)
        delete[] r3;
      if (polardet_n)
        delete[] polardet_n;
      if (polarang_n)
        delete[] polarang_n;

    } ///// End of calib leakage + polar

    if (rank == 0)
      printf("JUST BEFORE recalib \n");

    double *Ndtmp;
    double *relfI_n;
    // De-calibrate data and re-add dipole
    if (recalib || odip) {
      if (newrank != 0) {
        for (ii = 0; ii < (ns / 2 + 1) * ndet * nseg; ii++) {
          fdatas0[ii][0] += fodipole[ii][0];
          fdatas0[ii][1] += fodipole[ii][1];
        }
      }
    }

    if (recalib) {
      if (newrank != 0)
        for (iseg = 0; iseg < nseg; iseg++)
          for (idet = 0; idet < ndet; idet++)
            if (segon[idet * lnr + iseg + iseg_min])
              for (ii = 0; ii < ns / 2 + 1; ii++) {
                fdatas0[ii + idet * (ns / 2 + 1) + iseg * ndet * (ns / 2 + 1)]
                       [0] *= relCalib[idet + (iseg + iseg_min) * ndet];
                fdatas0[ii + idet * (ns / 2 + 1) + iseg * ndet * (ns / 2 + 1)]
                       [1] *= relCalib[idet + (iseg + iseg_min) * ndet];
              }

      relfI_n = new double[ndet * nsegtot];

      MPI_Barrier(MPI_COMM_WORLD);

      if (!recalLeak && !recalPolar) {
        if (newrank != 0)
          tmpinitmap = new double[npixtot];

        MPI_Bcast(tmpinitmap, npixtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }

      if (newrank != 0) {
        PNd = new double[nsmpixtot];
        Ndtmp = new double[ns];
      }

      for (idet = 0; idet < ndet; idet++)
        for (iseg = 0; iseg < nsegtot; iseg++)
          relCalib_old[idet + iseg * ndet] = relCalib[idet + iseg * ndet];

      absrecal = 1.0;
      int niterlim = 1;
      if (rloop < 6)
        niterlim = 50;

      for (int rerun = 0; rerun < niterlim; rerun++) {

        /// compute chi2
        chi2 = MPI_compute_chi2(
            fdatas0, fodipole, fdatas, fdata, tmpinitmap, nseg, datatopix,
            indpix, psip, relCalib, segon, SpN_all, npix, ns, ndet, iseg_min,
            lnr, nsubm, polar, polardet, polarang, GalBP, relfGal, newrank);
        if (rank == 0)
          printf("Before recal chi2 = %10.15g\n", chi2);

        zeros(relCalib, ndet * nsegtot);
        zeros(zeta, ndet * nsegtot);
        zeros(gamma, ndet * nsegtot);

        if (newrank != 0) {
          for (idet = 0; idet < ndet; idet++) {
            for (iseg = 0; iseg < nseg; iseg++) {
              if (segon[idet * lnr + iseg + iseg_min]) {

                double cPNPc = 0.0;
                double sPNd = 0.0;
                long pfd = 0;

                // Compute Ps
                deproject(tmpinitmap, indpix,
                          datatopix + iseg * ns + ns * nseg * idet,
                          psip + iseg * ns + ns * nseg * idet, ns, npix, Nd,
                          polar, *(polardet + idet), *(polarang + idet), GalBP,
                          *(relfGal + idet));

                fftplan = fftw_plan_dft_r2c_1d(ns, Nd, fdata, FFTW_ESTIMATE);
                fftw_execute(fftplan);
                fftw_destroy_plan(fftplan);

                for (ii = 0; ii < ns / 2 + 1; ii++) {
                  pfd = ii + idet * (ns / 2 + 1) + iseg * ndet * (ns / 2 + 1);
                  sPNd += (fdatas0[pfd][0] *
                               (fdata[ii][0] / absrecal + fodipole[pfd][0]) +
                           fdatas0[pfd][1] *
                               (fdata[ii][1] / absrecal + fodipole[pfd][1])) *
                          SpN_all[ii + idet * ns + idet * ndet * ns] /
                          double(nsubm) * 2.0;
                }

                for (ii = 0; ii < ns / 2 + 1; ii++) {
                  fdata[ii][0] =
                      (fdata[ii][0] / absrecal +
                       fodipole[ii + idet * (ns / 2 + 1) +
                                iseg * ndet * (ns / 2 + 1)][0]) *
                      sqrt(SpN_all[ii + idet * ns + idet * ndet * ns]);
                  fdata[ii][1] =
                      (fdata[ii][1] / absrecal +
                       fodipole[ii + idet * (ns / 2 + 1) +
                                iseg * ndet * (ns / 2 + 1)][1]) *
                      sqrt(SpN_all[ii + idet * ns + idet * ndet * ns]);
                }

                //// computation of cPNPc with an alternative approach
                for (ii = 0; ii < ns / 2 + 1; ii++)
                  cPNPc += (fdata[ii][0] * fdata[ii][0] +
                            fdata[ii][1] * fdata[ii][1]) /
                           2.0;

                // Compute final calib
                relCalib[idet + (iseg + iseg_min) * ndet] = sPNd / cPNPc;
                zeta[idet + (iseg + iseg_min) * ndet] = sPNd;
                gamma[idet + (iseg + iseg_min) * ndet] = cPNPc;
              }
            }
          }
        }

        // Merge information
        MPI_Reduce(relCalib, relfI_n, ndet * nsegtot, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        MPI_Bcast(relfI_n, ndet * nsegtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for (ii = 0; ii < ndet * nsegtot; ii++)
          relCalib[ii] = relfI_n[ii] / double(nsubm);

        MPI_Reduce(zeta, relfI_n, ndet * nsegtot, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        MPI_Bcast(relfI_n, ndet * nsegtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for (ii = 0; ii < ndet * nsegtot; ii++)
          zeta[ii] = relfI_n[ii];

        MPI_Reduce(gamma, relfI_n, ndet * nsegtot, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        MPI_Bcast(relfI_n, ndet * nsegtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for (ii = 0; ii < ndet * nsegtot; ii++)
          gamma[ii] = relfI_n[ii];

        if (recalibGlobal) {
          for (idet = 0; idet < ndet; idet++) {
            zeta_all[idet] = 0.0;
            gamma_all[idet] = 0.0;
            for (iseg = 0; iseg < nsegtot; iseg++) {
              if (segon[idet * lnr + iseg]) {
                zeta_all[idet] += zeta[idet + iseg * ndet];
                gamma_all[idet] += gamma[idet + iseg * ndet];
              }
            }
            for (iseg = 0; iseg < nsegtot; iseg++)
              relCalib[idet + iseg * ndet] = zeta_all[idet] / gamma_all[idet];

            if (rank == 0)
              printf("Coef calib det %ld is %10.15g\n", idet,
                     zeta_all[idet] / gamma_all[idet]);
          }
        }

        /// compute chi2
        chi2 = MPI_compute_chi2(
            fdatas0, fodipole, fdatas, fdata, tmpinitmap, nseg, datatopix,
            indpix, psip, relCalib, segon, SpN_all, npix, ns, ndet, iseg_min,
            lnr, nsubm, polar, polardet, polarang, GalBP, relfGal, newrank);
        if (rank == 0)
          printf("After recal chi2 = %10.15g\n", chi2);

        absrecal = 1.0;
        if (rloop < 6) {
          absrecal = 0.0;
          absrecal_old = 0.0;
          for (idet = 0; idet < ndet; idet++) {
            absrecal += zeta_all[idet] / gamma_all[idet] / double(ndet);
            absrecal_old += relCalib_old[idet] / double(ndet);
          }
          absrecal = absrecal / absrecal_old;
          if (rank == 0)
            printf("recal = %10.15g\n", absrecal);
        }
      }

      if (newrank != 0) {
        delete[] PNd;
        // delete [] Nd;
        delete[] Ndtmp;
        // delete [] fdata;
      }

      delete[] relfI_n;

      //************** Write recalibration coefficients to disk
      //****************//
      if (rank == 0) {
        // sprintf(mapoutstr, "%s/RelCalib%d",
        // Param->info_Ringnumb.groupName,rloop+21);
        FILE *fp;
        char filn2[100];
        sprintf(filn2, "%s%d", "relCalib.dat", rloop);
        fp = fopen(filn2, "w");
        fwrite(relCalib, ndet * nsegtot * sizeof(double), 1, fp);
        fclose(fp);
      }
    }

    /// Compute Chi2 and vary LFER parameters

    rloop += 1;

    if (!recalib && !recalLeak && !recalPolar)
      rloop = 0;

    if (newrank == 0) {
      delete[] PNdtot;
      delete[] PtNPmatStot;
      delete[] qtot;
    }
    delete[] d;
    delete[] tmpmap;
  }

  if (newrank == 0) {
    delete[] Mptot;
    delete[] hitstot;
  }

  //******************************************************************//
  //******************************************************************//
  //*********************** End of program ***************************//
  //******************************************************************//
  //******************************************************************//

  // MPI_Barrier(MPI_COMM_WORLD);
  // if (rank == 0)
  //   printf("DON'T FORGET e-3 and /20\n");

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    printf("!!!!!!!!!!!!!!!\n");

  MPI_Comm_free(&newcomm);
  MPI_Group_free(&newgroup);
  // MPI_Comm_free(&mastercomm);
  if (nsubm > 1)
    MPI_Group_free(&mastergroup);

  MPI_Finalize();

  // exit(0);
}

void write_reduced_map(char *mapoutstr, long nside, double *mapr,
                       long *indpix) {

  long ii;
  long NbWrite;
  double *map1d;
  map1d = new double[12 * nside * nside];

  // noise maps
  for (ii = 0; ii < 12 * nside * nside; ii++)
    map1d[ii] = 0.0;
  for (ii = 0; ii < 12 * nside * nside; ii++)
    if (indpix[ii] != -1) {
      if (indpix[ii] < 0)
        map1d[ii] = mapr[-indpix[ii] - 2];
      else
        map1d[ii] = mapr[indpix[ii]];
    }

  NbWrite = WriteMAP(map1d, mapoutstr, 8, 12 * nside * nside);
  if (NbWrite != 0)
    printf("WARNING, PROBLEM WITH WRITTEN DATA\n");

  delete[] map1d;
}

void write_tfAS(double *S, fftw_complex *fdatas, long *datatopix, long *indpix,
                double *psip, long npix, long ns, long ndet, long iseg,
                long iseg_min, long nseg, unsigned char *segon, long lnr,
                bool polar, double *polardet, double *polarang, int GalBP,
                double *relfGal, int idetr) {

  long idet1, ii;
  long countid = 0;
  double *Ps;
  double polard, polara, rfGal;

  fftw_plan fftplan;
  fftw_complex *fdata;

  Ps = new double[ns];
  fdata = new fftw_complex[ns / 2 + 1];

  long idetmin = 0;
  long idetmax = ndet - 1;
  if (idetr >= 0) {
    idetmin = idetr;
    idetmax = idetr;
  }

  for (idet1 = idetmin; idet1 <= idetmax; idet1++) {
    if (segon[idet1 * lnr + iseg + iseg_min]) {

      if (polar) {
        polard = polardet[idet1];
        polara = polarang[idet1];
      }
      if (GalBP != 0)
        rfGal = relfGal[idet1];

      deproject(S, indpix, datatopix + iseg * ns + ns * nseg * idet1,
                psip + iseg * ns + ns * nseg * idet1, ns, npix, Ps, polar,
                polard, polara, GalBP, rfGal);

      // Fourier transform of the data
      if (countid == 0)
        fftplan = fftw_plan_dft_r2c_1d(ns, Ps, fdata, FFTW_ESTIMATE);
      fftw_execute(fftplan);
      countid++;

      for (ii = 0; ii < ns / 2 + 1; ii++) {
        fdatas[ii + ns * idet1][0] = fdata[ii][0];
        fdatas[ii + ns * idet1][1] = fdata[ii][1];
      }
    }
  }
  if (countid)
    fftw_destroy_plan(fftplan);

  delete[] Ps;
  delete[] fdata;
}

void compute_ftrProcesdata(double *datas, fftw_complex *fdatas, long ns,
                           long ndet, long iseg, long nseg, long idetr) {

  // CAREFULL segon not implemented if this function needed

  long ii, idet1;
  double *data;

  fftw_plan fftplan;
  fftw_complex *fdata;

  data = new double[ns];
  fdata = new fftw_complex[ns / 2 + 1];

  long idetmin = 0;
  long idetmax = ndet - 1;
  if (idetr >= 0) {
    idetmin = idetr;
    idetmax = idetr;
  }

  for (idet1 = idetmin; idet1 <= idetmax; idet1++) {
    for (ii = 0; ii < ns; ii++)
      data[ii] = datas[ii + ns * iseg + idet1 * ns * nseg];

    // Fourier transform of the data
    if (idet1 == idetmin)
      fftplan = fftw_plan_dft_r2c_1d(ns, data, fdata, FFTW_ESTIMATE);
    fftw_execute(fftplan);
    if (idet1 == idetmax)
      fftw_destroy_plan(fftplan);

    for (ii = 0; ii < ns / 2 + 1; ii++) {
      fdatas[ii + ns * idet1][0] = fdata[ii][0];
      fdatas[ii + ns * idet1][1] = fdata[ii][1];
    }
  }

  delete[] data;
  delete[] fdata;
}

void do_PtNd(double *PNd, double *sumCross, fftw_complex *fdatas,
             long *datatopix, double *psip, double *SpN_all, long ns, long ndet,
             long *indpix, long ipix_min, long npix, bool CORRon, long iseg,
             long iseg_min, long nseg, unsigned char *segon, long lnr,
             double *relCalib, double *Mp, long *hits, bool polar,
             double *polardet, double *polarang, bool GalBP, double *relfGal,
             int idetr) {

  long ii, idet1, idet2, ppos, mpos;
  long countid = 0;

  double *Nk, *Nd;

  fftw_plan fftplan;
  fftw_complex *fdata, *Ndf;

  Nk = new double[ns / 2 + 1];
  fdata = new fftw_complex[ns / 2 + 1];
  Ndf = new fftw_complex[ns / 2 + 1];
  Nd = new double[ns];

  long idetmin = 0;
  long idetmax = ndet - 1;
  if (idetr >= 0) {
    idetmin = idetr;
    idetmax = idetr;
  }

  for (idet1 = idetmin; idet1 <= idetmax; idet1++) {

    if (segon[idet1 * lnr + iseg + iseg_min]) {
      // Init N-1d

      for (ii = 0; ii < ns / 2 + 1; ii++) {
        Ndf[ii][0] = 0;
        Ndf[ii][1] = 0;
      }

      for (idet2 = idetmin; idet2 <= idetmax; idet2++) {
        if (CORRon || idet2 == idet1) {

          if (segon[idet2 * lnr + iseg + iseg_min]) {

            for (ii = 0; ii < ns / 2 + 1; ii++) {
              fdata[ii][0] = fdatas[ii + idet2 * ns][0];
              fdata[ii][1] = fdatas[ii + idet2 * ns][1];
            }

            //****************** Cross power spectrum of the noise
            //***************//
            if (relCalib == NULL)
              for (ii = 0; ii < ns / 2 + 1; ii++)
                Nk[ii] = SpN_all[ii + idet1 * ns + idet2 * ndet * ns];
            else
              for (ii = 0; ii < ns / 2 + 1; ii++)
                Nk[ii] = SpN_all[ii + idet1 * ns + idet2 * ndet * ns] *
                         relCalib[idet1] * relCalib[idet2];

            //********************************* compute N^-1 d
            //***********************//
            for (ii = 0; ii < ns / 2 + 1; ii++) {
              Ndf[ii][0] += fdata[ii][0] * Nk[ii];
              Ndf[ii][1] += fdata[ii][1] * Nk[ii];
            }

            // Compute weight map for preconditioner
            if ((Mp != NULL) && (idet2 == idet1)) {

              // NEEDS TO BE IMPROVED FOR POLAR AND LEAKAGE HERE
              compute_diagPtNPCorr(Nk,
                                   datatopix + ns * iseg + ns * nseg * idet1,
                                   ns, indpix, ipix_min, npix, Mp);
              if (polar) {
                for (ii = 0; ii < npix; ii++) {
                  Mp[ii + npix] = Mp[ii] / 2.0;
                  Mp[ii + 2 * npix] = Mp[ii] / 2.0;
                  if (GalBP)
                    Mp[ii + 3 * npix] = Mp[ii] * relfGal[0] * relfGal[0];
                  // invert matrix:
                }

              } else if (GalBP)
                for (ii = 0; ii < npix; ii++)
                  Mp[ii + npix] = Mp[ii] * relfGal[0] * relfGal[0];
            }
          }
        }

      } // end of idet2 loop

      if (countid == 0)
        fftplan = fftw_plan_dft_c2r_1d(ns, Ndf, Nd, FFTW_ESTIMATE);
      fftw_execute(fftplan);
      countid++;

      for (ii = 0; ii < ns; ii++) {
        ppos = ii + ns * iseg + ns * nseg * idet1;
        mpos = indpix[datatopix[ppos]];
        if (mpos >= 0) {
          mpos = mpos - ipix_min;
          if (!polar) {
            PNd[mpos] += Nd[ii];
            if (GalBP)
              PNd[npix + mpos] += relfGal[idet1] * Nd[ii];
          } else {
            if (sumCross != NULL) {
              sumCross[mpos] += cos(2.0 * (psip[ppos] + polarang[idet1]));
              sumCross[npix + mpos] +=
                  sin(2.0 * (psip[ppos] + polarang[idet1]));
            } else {
              PNd[mpos] += Nd[ii];
              PNd[npix + mpos] += polardet[idet1] *
                                  cos(2.0 * (psip[ppos] + polarang[idet1])) *
                                  Nd[ii];
              PNd[2 * npix + mpos] +=
                  polardet[idet1] * sin(2.0 * (psip[ppos] + polarang[idet1])) *
                  Nd[ii];
              if (GalBP)
                PNd[3 * npix + mpos] += relfGal[idet1] * Nd[ii];
            }
          }
        }
      }

      // compute hit counts
      if (hits != NULL) {
        // printf("compute hits\n");
        for (ii = 0; ii < ns; ii++) {
          mpos = indpix[datatopix[ii + ns * iseg + ns * nseg * idet1]];
          if (mpos >= 0) {
            mpos = mpos - ipix_min;
            hits[mpos] += 1;
            if (polar)
              if (polardet[idet1] > 0.5)
                hits[npix + mpos] += 1;
          }
        }
      }
    }

  } // end of idet1 loop
  if (countid)
    fftw_destroy_plan(fftplan);

  delete[] Nk;
  delete[] fdata;
  delete[] Ndf;
  delete[] Nd;
}

void crosspar(double *PNd, long *datatopix, double *psip, long ns, long ndet,
              long *indpix, long ipix_min, long npix, long iseg, long iseg_min,
              long nseg, unsigned char *segon, long lnr, bool polar,
              double *polardet, double *polarang, int idetr) {

  long ii, idet1, ppos, mpos;

  long idetmin = 0;
  long idetmax = ndet - 1;
  if (idetr >= 0) {
    idetmin = idetr;
    idetmax = idetr;
  }

  for (idet1 = idetmin; idet1 <= idetmax; idet1++) {
    if (segon[idet1 * lnr + iseg + iseg_min]) {

      for (ii = 0; ii < ns; ii++) {
        ppos = ii + ns * iseg + ns * nseg * idet1;
        mpos = indpix[datatopix[ppos]];
        if (mpos >= 0) {
          mpos = mpos - ipix_min;
          if (!polar) {
            PNd[mpos] += 1;
          } else {
            PNd[mpos] += 1;
            PNd[npix + mpos] +=
                polardet[idet1] * cos(2.0 * (psip[ppos] + polarang[idet1]));
            PNd[2 * npix + mpos] +=
                polardet[idet1] * sin(2.0 * (psip[ppos] + polarang[idet1]));
            PNd[3 * npix + mpos] +=
                polardet[idet1] * cos(4.0 * (psip[ppos] + polarang[idet1]));
            PNd[4 * npix + mpos] +=
                polardet[idet1] * sin(4.0 * (psip[ppos] + polarang[idet1]));
          }
        }
      }
    }
  } // end of idet1 loop
}

void compute_diagPtNPCorr(double *Nk, long *datatopix, long ns, long *indpix,
                          long ipix_min, int npix, double *dPtNP) {

  long ii, k, kk, kk2, ipix, ii2;
  long *pixpos;
  long count, count_;
  long *pixtosamp;

  // fft stuff
  fftw_complex *Nk_;
  double *N_;
  fftw_plan fftplan;

  Nk_ = new fftw_complex[ns / 2 + 1];
  N_ = new double[ns];
  pixpos = new long[ns];

  // N^-1
  for (k = 0; k < ns / 2 + 1; k++) {
    Nk_[k][0] = abs(Nk[k]);
    Nk_[k][1] = 0.0;
    if (Nk[k] == 0.0)
      printf("Problem with input power spectrum Nk[%ld] = 0", k);
  }

  fftplan = fftw_plan_dft_c2r_1d(ns, Nk_, N_, FFTW_ESTIMATE);
  fftw_execute(fftplan);

  for (ii = 0; ii < ns; ii++)
    pixpos[ii] = indpix[datatopix[ii]];

  data_compare = new long[ns];
  pixtosamp = new long[ns];

  for (ii = 0; ii < ns; ii++)
    pixtosamp[ii] = ii;

  for (ii = 0; ii < ns; ii++)
    data_compare[ii] = pixpos[ii];

  qsort(pixtosamp, ns, sizeof(long), compare_global_array_long);
  qsort(data_compare, ns, sizeof(long), compare_long);

  long pstart = 0;
  while (data_compare[pstart] < 0)
    pstart++;

  count = pstart;

  for (ipix = data_compare[pstart]; ipix < npix + ipix_min; ipix++) {
    count_ = count;

    while ((count < ns) && (data_compare[count] == ipix))
      count++;

    if (count - count_ > 0) {
      for (ii = count_; ii < count; ii++) {
        ii2 = pixtosamp[ii];
        for (kk = count_; kk < count; kk++) {
          kk2 = pixtosamp[kk];
          if (abs(int(kk2 - ii2)) < ns / 2) {
            dPtNP[ipix - ipix_min] += N_[abs(int(ii2 - kk2))];
            // if (polar){
            //   dPtNP[ipix-ipix_min+npix] += cos(2*)*N_[abs(int(ii2-kk2))];
            //   dPtNP[ipix-ipix_min+2*npix] += sin(2*)*N_[abs(int(ii2-kk2))];
            //   dPtNP[ipix-ipix_min+3*npix] += cos(4*)*N_[abs(int(ii2-kk2))];
            //   dPtNP[ipix-ipix_min+4*npix] += sin(4*)*N_[abs(int(ii2-kk2))];
            // }
          }
        }
      }
    }
  }

  delete[] N_;
  delete[] Nk_;
  delete[] pixpos;
  delete[] pixtosamp;
  delete[] data_compare;

  // clean up
  fftw_destroy_plan(fftplan);
}

void ang2pix_ring(const long nside, double theta, double phi, long *ipix) {
  /*
    c=======================================================================
    c     gives the pixel number ipix (RING)
    c     corresponding to angles theta and phi
    c=======================================================================
  */

  int nl2, nl4, ncap, npix, jp, jm, ipix1;
  double z, za, tt, tp, tmp;
  int ir, ip, kshift;

  double piover2 = 0.5 * M_PI;
  double PI = M_PI;
  double twopi = 2.0 * M_PI;
  double z0 = 2.0 / 3.0;
  long ns_max = 8192;

  if (nside < 1 || nside > ns_max) {
    fprintf(stderr, "%s (%d): nside out of range: %ld\n", __FILE__, __LINE__,
            nside);
    exit(0);
  }

  if (theta < 0. || theta > PI) {
    fprintf(stderr, "%s (%d): theta out of range: %f\n", __FILE__, __LINE__,
            theta);
    exit(0);
  }

  z = cos(theta);
  za = fabs(z);
  if (phi >= twopi)
    phi = phi - twopi;
  if (phi < 0.)
    phi = phi + twopi;
  tt = phi / piover2; //  ! in [0,4)

  nl2 = 2 * nside;
  nl4 = 4 * nside;
  ncap = nl2 * (nside - 1); // ! number of pixels in the north polar cap
  npix = 12 * nside * nside;

  if (za <= z0) {

    jp = (int)floor(nside *
                    (0.5 + tt - z * 0.75)); /*index of ascending edge line*/
    jm = (int)floor(nside *
                    (0.5 + tt + z * 0.75)); /*index of descending edge line*/

    ir = nside + 1 + jp - jm; // ! in {1,2n+1} (ring number counted from z=2/3)
    kshift = 0;
    if ((ir & 1) == 0)
      kshift = 1; // ! kshift=1 if ir even, 0 otherwise

    ip = (int)floor((jp + jm - nside + kshift + 1) / 2) + 1; // ! in {1,4n}
    if (ip > nl4)
      ip = ip - nl4;

    ipix1 = ncap + nl4 * (ir - 1) + ip;
  } else {

    tp = tt - floor(tt); //      !MOD(tt,1.d0)
    tmp = sqrt(3. * (1. - za));

    jp = (int)floor(nside * tp * tmp);        // ! increasing edge line index
    jm = (int)floor(nside * (1. - tp) * tmp); // ! decreasing edge line index

    ir = jp + jm + 1; //        ! ring number counted from the closest pole
    ip = (int)floor(tt * ir) + 1; // ! in {1,4*ir}
    if (ip > 4 * ir)
      ip = ip - 4 * ir;

    ipix1 = 2 * ir * (ir - 1) + ip;
    if (z <= 0.) {
      ipix1 = npix - 2 * ir * (ir + 1) + ip;
    }
  }
  *ipix = ipix1 - 1; // ! in {0, npix-1}
}

void deproject(double *S, long *indpix, long *datatopix, double *psip, long ns,
               long npix, double *Ps, bool polar, double polard, double polara,
               int GalBP, double rfGal) {

  long ii, mpos;

  for (ii = 0; ii < ns; ii++) {
    mpos = indpix[datatopix[ii]];
    if (mpos == -1)
      printf("ERROR inside deproject mpos should always be > 0\n");
    if (mpos < 0) {
      mpos = -mpos - 2;
    }
    Ps[ii] = 0.0;
    if (!polar) {
      if (GalBP != -1)
        Ps[ii] = S[mpos];
      if (GalBP != 0)
        Ps[ii] += S[npix + mpos] * rfGal;
    } else {
      if (GalBP != -1)
        Ps[ii] = S[mpos] +
                 polard * (S[npix + mpos] * cos(2.0 * (psip[ii] + polara)) +
                           S[2 * npix + mpos] * sin(2.0 * (psip[ii] + polara)));
      if (GalBP != 0)
        Ps[ii] += S[3 * npix + mpos] * rfGal;
    }
  }
}

/// New CR
/*  deproject_partial

Compute the TOI from a map for ONE RING (ns samples) and ONE det using pointing
info.

Input :
double *S : the data map in compress format
long *indpix : the mapping from uncompress to compress pixel numbering
long *ringtopix : the pointing info (uncompress pixel numbering)
long ns : number of samples (size of ringtopix for one ring)

Output :
double *Ps : the output data read from the map using pointing info
*/
void deproject_partial(double *S, long *indpix, long *datatopix, long ns,
                       double *Ps) {
  long ii, mpos;

  for (ii = 0; ii < ns; ii++) {
    mpos = indpix[datatopix[ii]];
    if (mpos == -1)
      printf("ERROR inside deproject_partial mpos should always be > 0\n");
    if (mpos < 0)
      mpos = -mpos - 2;
    Ps[ii] = S[mpos];
  }
}

/*  compute_Nm1d

    Apply the noise spectrum weighting (using FFT) to timeline d
    i.e. computes N^(-1)Ps for ONE RING and ONE detector.

    Input :
    double *d : the input timeline
    long ns : number of samples in one ring
    double relcal : relative calib coef (used to recalibrate noise spectrum)
    double *SpN_all : the inverse noise spectrum (for the det and ring
   considered)

    Output :
    double *Nm1d : filled with N^(-1)d (real space) for ONE RING

*/
void compute_Nm1d(double *d, long ns, double *SpN_all, double relcal,
                  double *Nm1d) {
  long ii;

  relcal = 1.;

  // Compute fourier tranform of Ps vector
  fftw_plan fftplan;
  fftw_complex *tfd;

  tfd = new fftw_complex[ns / 2 + 1];

  fftplan = fftw_plan_dft_r2c_1d(ns, d, tfd, FFTW_ESTIMATE);
  fftw_execute(fftplan);
  fftw_destroy_plan(fftplan);

  // Apply noise weighting

  for (ii = 0; ii < ns / 2 + 1; ii++) {
    tfd[ii][0] *= (SpN_all[ii] * relcal * relcal);
    tfd[ii][1] *= (SpN_all[ii] * relcal * relcal);
  }

  // Compute inverse fourier tranform of tfNm1Ps
  fftplan = fftw_plan_dft_c2r_1d(ns, tfd, Nm1d, FFTW_ESTIMATE);
  fftw_execute(fftplan);
  fftw_destroy_plan(fftplan);

  delete[] tfd;
}

///////

void fMPI_merge(double *lmap, double *d, long nsmpix, long npix, long npixtot,
                int rank, int newrank, int size, int nsubm, bool polar,
                bool GalBP, MPI_Status status, bool allprc, MPI_Comm newcomm) {

  /// Each master node from subgroups send his submap to the master node. The
  /// master node contains all parts.
  // d is the input
  // lmap is the output

  long idet, ii;
  long ipixmin, ipixmax;
  long nns;

  if ((newrank == 0) && (rank != 0))
    MPI_Send(d, nsmpix, MPI_DOUBLE, 0, rank,
             MPI_COMM_WORLD); /// send submap to the master node

  if (rank == 0) {
    for (ii = 0; ii < nsmpix; ii++)
      lmap[ii] = d[ii];
    for (idet = 1; idet < nsubm; idet++) {
      ipixmin = idet * npix / nsubm;
      ipixmax = (idet + 1) * npix / nsubm;
      nns = ipixmax - ipixmin;
      MPI_Recv(lmap + ipixmin, nns, MPI_DOUBLE, idet * size / nsubm,
               idet * size / nsubm, MPI_COMM_WORLD, &status);
    } /// master node collects submaps
  }
  if (polar) {
    if ((newrank == 0) && (rank != 0))
      MPI_Send(d + nsmpix, nsmpix, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);

    if (rank == 0) {
      for (ii = 0; ii < nsmpix; ii++) {
        lmap[ii + npix] = d[ii + nsmpix];
        lmap[ii + 2 * npix] = d[ii + 2 * nsmpix];
      }
      for (idet = 1; idet < nsubm; idet++) {
        ipixmin = idet * npix / nsubm;
        ipixmax = (idet + 1) * npix / nsubm;
        nns = ipixmax - ipixmin;
        MPI_Recv(lmap + npix + ipixmin, nns, MPI_DOUBLE, idet * size / nsubm,
                 idet * size / nsubm, MPI_COMM_WORLD, &status);
      }
    }
    if ((newrank == 0) && (rank != 0)) {
      MPI_Send(d + 2 * nsmpix, nsmpix, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    }
    if (rank == 0)
      for (idet = 1; idet < nsubm; idet++) {
        ipixmin = idet * npix / nsubm;
        ipixmax = (idet + 1) * npix / nsubm;
        nns = ipixmax - ipixmin;
        MPI_Recv(lmap + 2 * npix + ipixmin, nns, MPI_DOUBLE,
                 idet * size / nsubm, idet * size / nsubm, MPI_COMM_WORLD,
                 &status);
      }
    if (GalBP) {
      if ((newrank == 0) && (rank != 0))
        MPI_Send(d + 3 * nsmpix, nsmpix, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
      if (rank == 0) {
        for (ii = 0; ii < nsmpix; ii++)
          lmap[ii + 3 * npix] = d[ii + 3 * nsmpix];
        for (idet = 1; idet < nsubm; idet++) {
          ipixmin = idet * npix / nsubm;
          ipixmax = (idet + 1) * npix / nsubm;
          nns = ipixmax - ipixmin;
          MPI_Recv(lmap + 3 * npix + ipixmin, nns, MPI_DOUBLE,
                   idet * size / nsubm, idet * size / nsubm, MPI_COMM_WORLD,
                   &status);
        }
      }
    }
  } else if (GalBP) {
    if ((newrank == 0) && (rank != 0))
      MPI_Send(d + nsmpix, nsmpix, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    if (rank == 0) {
      for (ii = 0; ii < nsmpix; ii++)
        lmap[ii + npix] = d[ii + nsmpix];
      for (idet = 1; idet < nsubm; idet++) {
        ipixmin = idet * npix / nsubm;
        ipixmax = (idet + 1) * npix / nsubm;
        nns = ipixmax - ipixmin;
        MPI_Recv(lmap + npix + ipixmin, nns, MPI_DOUBLE, idet * size / nsubm,
                 idet * size / nsubm, MPI_COMM_WORLD, &status);
      }
    }
  }

  if (allprc)
    MPI_Bcast(lmap, npixtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(lmap, npixtot, MPI_DOUBLE, 0, newcomm);
}

long WriteMAP(void *map1d, char *mapoutstr, int type, long nn) {

  int NbWrite = 0;

  FILE *fp;

  fp = fopen(mapoutstr, "w");
  fwrite(map1d, type, nn, fp);
  fclose(fp);

  return NbWrite;
}

long ReadVECT(void *data, string filename, long typesize, long imin,
              long imax) {

  ///// Read binary files

  FILE *fp;
  long status;

  char ffname[100];
  strcpy(ffname, filename.c_str());

  long nn = (imax - imin + 1);

  data = (void *)malloc(nn);

  fp = fopen(ffname, "r");

  /// Verify how to start from imin in the file
  status = fseek(fp, imin, SEEK_SET);

  fread(data, typesize, nn, fp);
  fclose(fp);

  return status;
}

long ReadLongVECT(long *data, string filename, long imin, long imax) {

  ///// Read binary files

  FILE *fp;
  long status;

  char ffname[300];
  strcpy(ffname, filename.c_str());

  long nn = (imax - imin + 1);

  fp = fopen(ffname, "r");

  /// Verify how to start from imin in the file
  status = fseek(fp, imin * sizeof(long), SEEK_SET);

  fread(data, sizeof(long), nn, fp);
  printf("hello %d\n", __LINE__);
  fclose(fp);

  return status;
}

long ReadDoubleVECT(double *data, string filename, long imin, long imax) {

  ///// Read binary files

  FILE *fp;
  long status;

  char ffname[300];
  strcpy(ffname, filename.c_str());

  long nn = (imax - imin + 1);

  fp = fopen(ffname, "r");

  /// Verify how to start from imin in the file
  status = fseek(fp, imin * sizeof(double), SEEK_SET);

  fread(data, sizeof(double), nn, fp);
  fclose(fp);

  return status;
}

void read_bolofile(string fname, list<string> &bolos) {
  char buff[256];
  string line;

  ifstream BOLO(fname.c_str());
  if (!BOLO.is_open()) {
    cerr << "Error opening bolometer file '" << fname << "'. Exiting.\n";
    exit(1);
  }

  while (!BOLO.eof()) {
    BOLO.getline(buff, 255);
    line = buff;

    line.erase(0, line.find_first_not_of(" \t")); // remove leading white space
    if (line.empty() || line[0] == '#')
      continue; // skip if empty or commented
    line = line.substr(0, line.find_first_of(" \t")); // pick out first word

    bolos.push_back(line);
  }

  BOLO.close();
}

void read_bolo_offsets(string bolo, string file_BoloOffsets, double *offsets) {

  double lel, xel, angl;
  long num, pnum;
  int nobolo = 1;

  char boloname[100];
  char tempchar[100];
  FILE *fp;

  printf("%s\n", file_BoloOffsets.c_str());

  if ((fp = fopen(file_BoloOffsets.c_str(), "r")) == NULL) {
    cerr << "ERROR: Can't find offset file. Exiting. \n";
    exit(1);
  }
  while (fscanf(fp, "%ld%ld%lf%lf%lf%s%s\n", &num, &pnum, &lel, &xel, &angl,
                tempchar, boloname) != EOF) {
    if (bolo == boloname) {
      nobolo = 0;
      offsets[0] = xel;
      offsets[1] = lel;
      offsets[2] = angl;
    }
  }
  fclose(fp);

  if (nobolo) {
    cerr << "Bolometer name not found in offset list" << endl;
    exit(1);
  }
}

void fmodelTC(double *pp, long ns, fftw_complex *Fu) {

  long ii;

  double tau = pp[0];
  double ampl = pp[1];
  // double shs = pp[2];
  // double *timetc;
  double kk2;
  for (ii = 0; ii < ns / 2 + 1; ii++) {
    kk2 = (double)ii * (double)ii * 4.0 * M_PI * M_PI / double(ns) / double(ns);
    Fu[ii][0] = (1.0 - ampl + ampl / (1.0 + kk2 * tau * tau));
    Fu[ii][1] = -ampl * (double)ii * tau / (1.0 + kk2 * tau * tau);
  }
}

double MPI_compute_chi2(fftw_complex *fdatas0, fftw_complex *fodipole,
                        fftw_complex *fdatas, fftw_complex *fdata, double *map,
                        long nseg, long *datatopix, long *indpix, double *psip,
                        double *relCalib, unsigned char *segon, double *SpN_all,
                        long npix, long ns, long ndet, long iseg_min, long lnr,
                        int nsubm, bool polar, double *polardet,
                        double *polarang, bool GalBP, double *relfGal,
                        long newrank) {

  long ii, iseg, idet, posrelC, posfdatas;
  double chi2 = 0.0;
  double chi2tot = 0.0;

  if (newrank != 0) {
    for (iseg = 0; iseg < nseg; iseg++) {

      write_tfAS(map, fdatas, datatopix, indpix, psip, npix, ns, ndet, iseg,
                 iseg_min, nseg, segon, lnr, polar, polardet, polarang, GalBP,
                 relfGal, -1);

      //// compute chi2
      for (idet = 0; idet < ndet; idet++) {

        if (segon[idet * lnr + iseg + iseg_min]) {
          // compute first term of chi2
          for (ii = 0; ii < ns / 2 + 1; ii++) {
            posrelC = idet + (iseg + iseg_min) * ndet;
            posfdatas = ii + idet * (ns / 2 + 1) + iseg * ndet * (ns / 2 + 1);
            fdata[ii][0] = (fdatas0[posfdatas][0] -
                            relCalib[posrelC] * (fdatas[ii + idet * ns][0] +
                                                 fodipole[posfdatas][0])) *
                           sqrt(SpN_all[ii + idet * ns + idet * ndet * ns]);
            fdata[ii][1] = (fdatas0[posfdatas][1] -
                            relCalib[posrelC] * (fdatas[ii + idet * ns][1] +
                                                 fodipole[posfdatas][1])) *
                           sqrt(SpN_all[ii + idet * ns + idet * ndet * ns]);
          }

          for (ii = 0; ii < ns / 2 + 1; ii++)
            chi2 +=
                (fdata[ii][0] * fdata[ii][0] + fdata[ii][1] * fdata[ii][1]) /
                double(nsubm) * 2.0;
        }
      }

    } // end of iseg loop
  }

  // Chi2:
  MPI_Reduce(&chi2, &chi2tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&chi2tot, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  chi2 = chi2tot;

  return chi2;
}

void forceLeakzero(double *PtNPmatStot, long nside, long *indpix, long ipixmin,
                   long nsmpix, bool polar) {
  /////////// force leakage to be zero at high galactic latitude (7/8)
  long ii;

  for (ii = 0; ii < 12 * nside * nside; ii++)
    if (((ii < 12 * nside * nside / 8) || (ii > 12 * nside * nside * 7 / 8)) &&
        (indpix[ii] >= 0)) {
      if (polar)
        PtNPmatStot[indpix[ii] - ipixmin + 3 * nsmpix] = 0;
      else
        PtNPmatStot[indpix[ii] - ipixmin + nsmpix] = 0;
    }
}

void zeros(double *data, long nn) {
  for (int ii = 0; ii < nn; ii++)
    data[ii] = 0;
}

void initarray(double *data, long nn, double val) {
  for (int ii = 0; ii < nn; ii++)
    data[ii] = val;
}

////////////////// some projection functions

void slaDs2tp(double ra, double dec, double raz, double decz, double *xi,
              double *eta, int *j)
/*
**  - - - - - - - - -
**   s l a D s 2 t p
**  - - - - - - - - -
**
**  Projection of spherical coordinates onto tangent plane
**  ('gnomonic' projection - 'standard coordinates').
**
**  (double precision)
**
**  Given:
**     ra,dec      double   spherical coordinates of point to be projected
**     raz,decz    double   spherical coordinates of tangent point
**
**  Returned:
**     *xi,*eta    double   rectangular coordinates on tangent plane
**     *j          int      status:   0 = OK, star on tangent plane
**                                    1 = error, star too far from axis
**                                    2 = error, antistar on tangent plane
**                                    3 = error, antistar too far from axis
**
**  Last revision:   18 July 1996
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
#define TINY 1e-6
{
  double sdecz, sdec, cdecz, cdec, radif, sradif, cradif, denom;

  /* Trig functions */
  sdecz = sin(decz);
  sdec = sin(dec);
  cdecz = cos(decz);
  cdec = cos(dec);
  radif = ra - raz;
  sradif = sin(radif);
  cradif = cos(radif);

  /* Reciprocal of star vector length to tangent plane */
  denom = sdec * sdecz + cdec * cdecz * cradif;

  /* Handle vectors too far from axis */
  if (denom > TINY) {
    *j = 0;
  } else if (denom >= 0.0) {
    *j = 1;
    denom = TINY;
  } else if (denom > -TINY) {
    *j = 2;
    denom = -TINY;
  } else {
    *j = 3;
  }

  /* Compute tangent plane coordinates (even in dubious cases) */
  *xi = cdec * sradif / denom;
  *eta = (sdec * cdecz - cdec * sdecz * cradif) / denom;
}

void slaDtp2s(double xi, double eta, double raz, double decz, double *ra,
              double *dec)
/*
**  - - - - - - - - -
**   s l a D t p 2 s
**  - - - - - - - - -
**
**  Transform tangent plane coordinates into spherical.
**
**  (double precision)
**
**  Given:
**     xi,eta      double   tangent plane rectangular coordinates
**     raz,decz    double   spherical coordinates of tangent point
**
**  Returned:
**     *ra,*dec    double   spherical coordinates (0-2pi,+/-pi/2)
**
**  Called:  slaDranrm
**
**  Last revision:   3 June 1995
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
  double sdecz, cdecz, denom;

  sdecz = sin(decz);
  cdecz = cos(decz);
  denom = cdecz - eta * sdecz;
  *ra = slaDranrm(atan2(xi, denom) + raz);
  *dec = atan2(sdecz + eta * cdecz, sqrt(xi * xi + denom * denom));
}

double slaDranrm(double angle)
/*
**  - - - - - - - - - -
**   s l a D r a n r m
**  - - - - - - - - - -
**
**  Normalize angle into range 0-2 pi.
**
**  (double precision)
**
**  Given:
**     angle     double      the angle in radians
**
**  The result is angle expressed in the range 0-2 pi (double).
**
**  Defined in slamac.h:  D2PI, dmod
**
**  Last revision:   19 March 1996
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
  double w;

  w = dmod(angle, D2PI);
  return (w >= 0.0) ? w : w + D2PI;
}

void pointingshift(double x, double y, long nn, double *thetap, double *phip,
                   double *psip, double *theta, double *phi, double *psi) {

  long ii, jj, kk;
  double theta_, phi_, aaa, rrr, psitmp;
  double *drp, *dr, *eph;

  theta_ = asin(sqrt(x * x + y * y));
  phi_ = atan2(y, x);
  for (ii = 0; ii < nn; ii++) {
    theta[ii] = acos(sin(thetap[ii]) * sin(theta_) * cos(phi_ + psip[ii]) +
                     cos(thetap[ii]) * cos(theta_));
    phi[ii] = atan2(-sin(theta_) * sin(phi_ + psip[ii]),
                    (-cos(thetap[ii]) * sin(theta_) * cos(phi_ + psip[ii]) +
                     sin(thetap[ii]) * cos(theta_))) +
              phip[ii];
  }

  drp = new double[3];
  dr = new double[3];
  double R[3][3];
  eph = new double[3];

  drp[2] = -sin(theta_) * cos(phi_);
  for (ii = 0; ii < 3; ii++)
    for (jj = 0; jj < 3; jj++)
      R[ii][jj] = 0.0;
  R[1][1] = -1.0;
  eph[2] = 0.0;

  for (ii = 0; ii < nn; ii++) {
    drp[0] = cos(psip[ii]) * cos(theta_);
    drp[1] = sin(psip[ii]) * cos(theta_);
    R[0][0] = -cos(thetap[ii]);
    R[0][2] = sin(thetap[ii]);
    R[2][0] = sin(thetap[ii]);
    R[2][2] = cos(thetap[ii]);
    for (kk = 0; kk < 3; kk++)
      dr[kk] = 0.0;
    for (jj = 0; jj < 3; jj++)
      for (kk = 0; kk < 3; kk++)
        dr[kk] += R[kk][jj] * drp[jj];
    eph[0] = -sin(phi[ii] - phip[ii]);
    eph[1] = cos(phi[ii] - phip[ii]);
    aaa = 0.0;
    rrr = 0.0;
    for (kk = 0; kk < 3; kk++) {
      rrr += dr[kk] * dr[kk];
      aaa += dr[kk] * eph[kk];
    }
    aaa = aaa / sqrt(rrr);
    if (aaa > 1.0)
      aaa = 1.0;
    if (aaa < -1.0)
      aaa = -1.0;
    psitmp = acos(aaa);
    if (dr[2] < 0)
      psitmp = -psitmp;
    psitmp = (fmod(psitmp + D2PI / 4.0 + 2.0 * D2PI, D2PI)) - D2PI / 2.0;
    psi[ii] = psitmp;
  }
  delete[] drp;
  delete[] dr;
  delete[] eph;
}
