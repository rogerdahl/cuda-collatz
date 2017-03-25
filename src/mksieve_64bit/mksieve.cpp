// Sieve Generator for Collatz Delay Record Calculator
// Submitted to the pulic domain. Roger Dahl, 2008
// Thanks to Eric Roosendaal for describing the optimizations

// The sieve checks whether paths come together. If two paths join then the
// upper one can never yield a class record and can be skipped.
//
// Example: Any N of the form 8k+4 will reach 6k+4 after 3 steps, and so will
// any N of the form 8k+5. Therefore no N of the form 8k+5 can be a class record
// (5 itself being the only exception). So any N of the form 8k+5 does not need
// to be checked, and all positions of the form 8k+5 in the sieve contain a
// zero.

// See generate_sieve_simple() below to see how the sieve bits are generated for
// small sieve sizes. To generate a sieve, say 10 bits wide, 1024k + i is
// calculated, where i loops from 0 to 1023. 10 steps of x/2 or (3x+1)/2 are
// done. After that we get a number on the form 3^p + r. If some of those
// numbers end up with the same p and r, all of them can be skipped, except the
// lowest one.
//
// The sieve is applied by setting index = N mod 1024 and then using the index
// to look up the disable calculation bit in the sieve table. The sieve can only
// be used for N higher than the sieve size, 1024 in this case.
//
// To generate larger sieves, we must use a more advanced technique. This
// technique is implemented in generate_sieve(). It requires a large amount of
// external storage (1.2TB for a 2^35 sieve). The external storage is handled by
// STXXL.
//
// STXXL and memory usage:
//
// When reading the STXXL docs, M is the size of main memory B is the block size
// in bytes D is the number of disks N is the input size measured in bytes.
//
// To generate as many bits as can be stored in 2GB of memory, we need 2
// (because we remove even numbered bits in the end) * 2GB * 8 (bits per byte) =
// 32gbit = 2^35 = 34 359 738 368 bits. Which means we need to create the same
// amount of Collatz records. Each record is 16 bytes (using bitfields). 16
// bytes * 34 359 738 368 records = 512GB.
//
// The stxxl::sort is not in-place. It requires N bytes of external memory to
// store the sorted runs during sorting. That means that we need another 512GB
// to sort the Collatz records. 2 * 512GB = 1TB.
//
// After sorting, the 512GB used while sorting is freed up.
//
// We then create the same number of index records, each 64 bits + 8 bits = 9
// bytes. This is less than the extra memory used by sorting the Collatz
// records, so the peak disk usage is 1TB.
//
// There might be some overhead as well, so we need approximately 1.2TB of disk
// space to generate a 2^35 sieve.

#include "stdafx.h"

using namespace std;
using namespace boost;
using namespace stxxl;
namespace po = boost::program_options;

// Bytes of memory to use when sorting.
const int sort_bufsize(1024 * 1024 * 1024);

// Collatz record.

// max values for sieve_size 2^33
// p: 0x21 (100001)
// r: 0x13BFEFA65ABB82
// i: 0x200000000
//
// max values for sieve_bits 2^35
// p: 0x23 = 100011 = 6 bits
// r: 0xB1 BF6C D930 979A = 10110001 10111111 01101100 11011001 00110000
// 10010111 10011010 = 56 bits
// i: 0x8 0000 0000       =                       1000 00000000 00000000
// 00000000 00000000 = 36 bits

struct collatz_record
{
  collatz_record(u64 ap, u64 ar, u64 ai) : p(ap), r(ar), i(ai)
  {
  }
  collatz_record() : p(0), r(0), i(0)
  {
  }

  // p=6, r=56 and i=36 is enough for sieve_bits 2^35. We use some more bits
  // while keeping the full size at 16 bytes in the hopes of better performance.
  u64 r;
  u64 i : 56;
  u64 p : 8;
};

// Sort Collatz record by p, r, index.

bool operator<(const collatz_record& a, const collatz_record& b)
{
  if (a.p < b.p) {
    return true;
  }
  if (a.p > b.p) {
    return false;
  }

  if (a.r < b.r) {
    return true;
  }
  if (a.r > b.r) {
    return false;
  }

  if (a.i < b.i) {
    return true;
  }
  if (a.i > b.i) {
    return false;
  }

  return false;
}

bool operator==(const collatz_record& a, const collatz_record& b)
{
  return a.p == b.p && a.r == b.r && a.i == b.i;
}

bool operator!=(const collatz_record& a, const collatz_record& b)
{
  return !(a == b);
}

struct cmp : public std::less<collatz_record>
{
  collatz_record min_value() const
  {
    return collatz_record(0, 0, 0);
  }
  collatz_record max_value() const
  {
    return collatz_record(
        0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff);
  }
};

// Index record.

struct index_record
{
  index_record(u64 ai, bool ac) : i(ai), c(ac)
  {
  }
  index_record() : i(0), c(false)
  {
  }
  u64 i;
  bool c;
};

// Sort index record by index.

bool operator<(const index_record& a, const index_record& b)
{
  return a.i < b.i;
}

bool operator==(const index_record& a, const index_record& b)
{
  return a.i == b.i;
}

bool operator!=(const index_record& a, const index_record& b)
{
  return !(a == b);
}

struct cmp2 : public std::less<index_record>
{
  index_record min_value() const
  {
    return index_record(0, false);
  }
  index_record max_value() const
  {
    return index_record(0xffffffffffffffff, false);
  }
};

bool generate_sieve(u64 sieve_bits)
{
  const u64 sieve_size(1LL << sieve_bits);
  try {
    boost::timer time_operation;
    boost::timer time_display;
    boost::timer time_total;

    stxxl::vector<collatz_record> collatz;
    collatz.reserve(sieve_size);

    // Size
    wcout << wformat(L"Size of Collatz record: %d bytes")
                 % sizeof(collatz_record)
          << endl;
    wcout << wformat(L"Size of Collatz vector: %d bytes")
                 % (sizeof(collatz_record) * sieve_size)
          << endl;

    // Generate the sieve table.

    time_operation.restart();
    time_display.restart();
    for (u64 i(0); i < sieve_size; ++i) {
      // Display status.
      if (time_display.elapsed() > 1.0f || i == sieve_size - 1) {
        time_display.restart();
        wcout << wformat(L"Generating Collatz records %d / %d (%.1fs)\r")
                     % (i + 1) % sieve_size % time_operation.elapsed()
              << flush;
      }
      // Calculate p and r for this index.
      u64 p(0);
      u64 r(i);
      for (u64 j(sieve_bits); j; --j) {
        if (r & 1) {
          ++p;
          r = (r << 1) + r + 1; // 3 * r + 1;
        }
        r >>= 1; // r / 2
      }
      collatz.push_back(collatz_record(p, r, i));
    }
    wcout << endl;

    // Sort Collatz records.

    wcout << L"Sorting Collatz records... " << flush;
    time_operation.restart();
    stxxl::sort(collatz.begin(), collatz.end(), cmp(), sort_bufsize);
    wcout << wformat(L"Completed (%.1fs)") % time_operation.elapsed() << endl;

    // Generate index records from Collatz records.

    stxxl::vector<index_record> indexes;
    indexes.reserve(sieve_size);
    u64 last_p(-1);
    u64 last_r(-1);
    u64 cnt_mark(1);
    time_operation.restart();
    time_display.restart();
    for (stxxl::vector<collatz_record>::const_iterator iter(collatz.begin());
         iter != collatz.end(); ++iter) {
      if (time_display.elapsed() > 1.0f || cnt_mark == sieve_size) {
        time_display.restart();
        wcout << wformat(L"Generating index records %d / %d (%.1fs)\r")
                     % cnt_mark % sieve_size % time_operation.elapsed()
              << flush;
      }

      bool c(true);
      if (iter->p == last_p && iter->r == last_r) {
        c = false;
      }
      else {
        last_p = iter->p;
        last_r = iter->r;
      }
      indexes.push_back(index_record(iter->i, c));
      ++cnt_mark;
    }
    wcout << endl;

    // Clear the Collatz vector to save room.

    wcout << wformat(L"Clearing Collatz records... ") << flush;
    time_operation.restart();
    collatz.clear();
    wcout << wformat(L"Completed (%.1fs)") % time_operation.elapsed() << endl;

    // Sort index/calculate records.

    wcout << L"Sorting index records... " << flush;
    time_operation.restart();
    stxxl::sort(indexes.begin(), indexes.end(), cmp2(), sort_bufsize);
    wcout << wformat(L"Completed (%.1fs)") % time_operation.elapsed() << endl;

    // Write sieve bits to file. Do not write the bits with even indexes because
    // we don't check even numbers. I checked if the same result could be
    // created by not generating even numbered Collatz records in the first
    // place, but that is NOT the same.

    time_operation.restart();
    ofstream out_bits(
        str(wformat(L"sieve_%d.dat") % sieve_bits).c_str(), ios::binary);
    u64 bits(0);
    u64 word(0);
    u64 cnt_create(0);
    time_display.restart();
    for (stxxl::vector<index_record>::const_iterator iter(indexes.begin());
         iter != indexes.end(); ++iter) {
      // Skip all even N.
      if (!(iter->i & 1)) {
        continue;
      }
      // Status.
      if (time_display.elapsed() > 1.0f || cnt_create == sieve_size - 1) {
        time_display.restart();
        wcout << wformat(L"Creating sieve data file %d / %d (%.1fs)\r")
                     % cnt_create % sieve_size % time_operation.elapsed()
              << flush;
      }
      // Add bit to current word.
      if (iter->c) {
        word |= (1LL << ((sizeof(word) * 8) - 1));
      }
      // Write word to disk if complete.
      if (++bits == sizeof(word) * 8) {
        out_bits.write(reinterpret_cast<char*>(&word), sizeof(word));
        bits = 0;
      }
      word >>= 1;
      // Status.
      cnt_create += 2;
    }
    wcout << endl;
    // Warn if last word was not full.
    if (bits) {
      wcout << L"Warning: sieve_size not divisible by 64" << endl;
      out_bits.write(reinterpret_cast<char*>(&word), sizeof(word));
    }
    wcout << wformat(L"Total time %.1fs") % time_total.elapsed() << endl;
  } catch (const std::exception& ex) {
    wcout << L"Exception: " << ex.what() << endl;
    return false;
  } catch (...) {
    wcout << L"Unknown exception" << endl;
    return false;
  }
  return true;
}

// This function generates the same data as generate_sieve(). It makes it easy
// to see what's actually going on but can only be used to generate small sieves
// because of the "painter's algorithm" where it keeps going back over all
// previously generated records for each new record.
void generate_sieve_simple(u64 sieve_bits)
{
  const u64 sieve_size(1LL << sieve_bits);
  // Generate the Collatz records.
  scoped_array<u64> p(new u64[sieve_size]);
  scoped_array<u64> r(new u64[sieve_size]);
  for (int i(0); i < sieve_size; ++i) {
    p[i] = 0;
    r[i] = i;
    for (int j(0); j < sieve_bits; ++j) {
      if (r[i] & 1) {
        ++p[i];
        r[i] = 3 * r[i] + 1;
      }
      r[i] /= 2;
    }
  }
  // For each record, check all lower records to see if the record exists. If a
  // lower one exists, disable calculation for the high record.
  //
  // We're not storing the sieve_bits, just keeping stats.
  u64 disabled_calculations(0);
  for (int i(0); i < sieve_size; ++i) {
    bool sieve(true);
    for (int j(i - 1); j >= 0; --j) {
      if (p[i] == p[j] && r[i] == r[j]) {
        sieve = false; // This would be stored in a full implementation.
        ++disabled_calculations;
        break;
      }
    }
    // Display percentage of disabled calculations if we're at a power-of-two
    // number.
    if (!((i - 1) & i)) {
      u64 v(i);
      u64 pos(0);
      while (v >>= 1) {
        ++pos;
      }
      wcout << wformat(L"2^%d - %.2f%%") % pos
                   % (double(disabled_calculations) / (double(i)) * 100)
            << endl;
    }
  }
}

// Display the bits in a sieve_bits file.
void display_bits(u64 sieve_bits)
{
  ifstream in(str(wformat(L"sieve_%d.dat") % sieve_bits).c_str(), ios::binary);
  if (!in.good()) {
    return;
  }
  u64 i(0);
  while (!in.eof()) {
    u64 word;
    in.read(reinterpret_cast<char*>(&word), sizeof(word));
    for (u64 j(0); j < sizeof(word) * 8; ++j) {
      wcout << i << L" " << (word & 1) << endl;
      word >>= 1;
      ++i;
    }
  }
}

// Get the percentage of disabled calculations in a sieve_bits file.
double get_disabled_calculations(u64 sieve_bits)
{
  ifstream in(str(wformat(L"sieve_%d.dat") % sieve_bits).c_str(), ios::binary);
  if (!in.good()) {
    throw int(0);
  }
  u64 disabled_calculations(0);
  u64 i(0);
  while (!in.eof()) {
    u64 word;
    in.read(reinterpret_cast<char*>(&word), sizeof(word));
    for (u64 j(0); j < sizeof(word) * 8; ++j) {
      if (!(word & 1)) {
        ++disabled_calculations;
      }
      word >>= 1;
    }
    i += sizeof(word) * 8;
  }
  return double(disabled_calculations) / double(i) * 100;
}

// Find the maximum number of bits used for a given sieve_size.
void find_max(u64 sieve_bits)
{
  const u64 sieve_size(1LL << sieve_bits);
  s64 p_max(0);
  s64 r_max(0);
  boost::timer time_display;
  boost::timer time_operation;

  for (u64 i(0); i < sieve_size; ++i) {
    // Calculate p and r for this index.
    s64 p(0);
    s64 r(i);
    for (u64 j(sieve_bits); j; --j) {
      if (r & 1) {
        ++p;
        r = (r << 1) + r + 1; // 3 * r + 1;
      }
      r >>= 1; // r / 2
      // Make sure r doesn't overflow.
      if (r < 0) {
        wcout << L"r overflow at " << i << endl;
      }
    }
    if (p > p_max) {
      p_max = p;
    }
    if (r > r_max) {
      r_max = r;
    }
  }
  wcout << wformat(L"max for sieve_size %x: p: %x, r: %x") % sieve_size % p_max
               % r_max
        << endl;
}

// Get Delay of N with only the most basic optimizations.
u64 D(u64 n)
{
  u64 delay(0);
  while (n > 1) {
    if (n & 1) {
      n = (n << 1) + n + 1; // 3 * n + 1;
    }
    else {
      n >>= 1; // n / 2
    }
    ++delay;
  }
  return delay;
}

// Generate table of percentages of disabled calculations for sieve sizes up to
// sieve_bits.
void gen_percentage_table(u64 sieve_bits)
{
  const u64 first(8);

  std::vector<double> dis;
  for (int i(first); i <= sieve_bits; ++i) {
    wcout << L">>> " << i << endl;
    generate_sieve(i);
    dis.push_back(get_disabled_calculations(sieve_bits));

    u64 cnt(first);
    for (std::vector<double>::iterator iter(dis.begin()); iter != dis.end();
         ++iter) {
      wcout << wformat(L"2^%d - %.2f%%") % cnt % *iter << endl;
      ++cnt;
    }
  }
}

int wmain(int ac, wchar_t** av)
{
  // switch from C locale to user's locale
  std::locale::global(std::locale("")); // will be used for all new streams
  wcout.imbue(locale("")); // this stream was already created, so must imbue

  wcout << L"Sieve Generator for Collatz Delay Record Calculator" << endl;
  wcout << L"Submitted to the pulic domain. Roger Dahl, 2008" << endl;
  wcout << L"Thanks to Eric Roosendaal for describing the optimizations"
        << endl;

  int sieve_bits;

  // Handle program options.
  try {
    po::options_description desc("Options");
    po::variables_map vm;
    desc.add_options()("help,h", "Produce help message")(
        "size,s", po::value<int>(&sieve_bits)->default_value(20),
        "Set the size of the sieve (2^X)");

    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }
  } catch (exception& e) {
    wcout << L"Error: Couldn't parse command line arguments: " << e.what()
          << endl;
    return 1;
  }

  generate_sieve(sieve_bits);

  // display_bits_simple(sieve_bits);
  // gen_percentage_table(sieve_bits);
  // display_bits(sieve_bits);
  // generate_sieve_simple(sieve_bits);
  // display_bits_simple(sieve_bits);

  return 0;
}
