// Collatz Delay Record Calculator Reference Implementation
// Submitted to the pulic domain. Roger Dahl, 2008
// Thanks to Eric Roosendaal for describing the optimizations.
//
// Reference implementation that verifies the high level optimizations used in
// the GPU version in C++

#include "stdafx.h"

using namespace std;
using namespace boost;
namespace po = boost::program_options;

// Get Delay of N with only the most basic optimizations.
u32 D(u64 n)
{
  u32 delay(0);
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

// Find the highest bit set in a word.
u32 high_bit(u64 w)
{
  u32 ret(1);
  while (w >>= 1) {
    ++ret;
  }
  return ret;
}

// Optimized Collatz calculator.
void collatz_ref(u32 step_bits, u32 tail_bits, bool benchmark)
{
  // Optimization: We use a table, called a sieve, of bits that disable
  // calculation for a big percentage of N. The sieve data is generated by a
  // separate program and we just load it here. See mksieve.cpp for details on
  // how the sieve works and how it is generated.
  u64 sieve_size(0);
  u64 sieve_words(0);
  vector<u64> sieve_data;
  wcout << L"Loading sieve" << endl;
  ifstream sieve_file(L".\\sieve.dat");
  if (!sieve_file.good()) {
    wcout << L"Warning: Couldn't open sieve.dat. Calculations will be slower"
          << endl;
  }
  else {
    sieve_file.seekg(0, ios::end);
    u64 sieve_bytes(static_cast<u64>(sieve_file.tellg()));
    sieve_words = sieve_bytes / sizeof(u64);
    sieve_size = sieve_words * sizeof(u64) * 8;
    sieve_data.resize(sieve_words);
    sieve_file.seekg(0, ios::beg);
    sieve_file.read(reinterpret_cast<char*>(&sieve_data[0]), sieve_bytes);
    sieve_file.close();
    wcout << wformat(L"Loaded %d entry sieve (%d bytes)") % sieve_size
                 % sieve_bytes
          << endl;
  }
  const u64 sieve_mask(sieve_words - 1);

  // Optimization: A precalculation allows us to jump ahead k steps on each
  // iteration. We break up the current N into two parts, b (the k least
  // significant bits, interpreted as an integer), and a (the rest of the bits
  // as an integer). The result of jumping ahead k+c[b] steps can be found as:
  //
  // f k+c[b](a 2k+b) = a 3c[b]+d[b]
  //
  // The c and d arrays are calculated for all possible k-bit numbers a, where
  // d[a] is the result of applying the f function k times to b, and c[a] is the
  // number of odd numbers encountered on the way.
  //
  // For a 5 bit step_size, the c and d tables will look like this:
  //
  // c [0...31] =
  // {0,3,2,2,2,2,2,4,1,4,1,3,2,2,3,4,1,2,3,3,1,1,3,3,2,3,2,4,3,3,4,5}
  //
  // d [0...31] =
  // {0,2,1,1,2,2,2,20,1,26,1,10,4,4,13,40,2,5,17,17,2,2,20,20,8,22,8,71,26,26,80,242}
  //
  // For more inwformation about this optimization see
  // http://en.wikipedia.org/wiki/Collatz_conjecture#As_a_parity_sequence
  const u32 step_size(1LL << step_bits);
  const u64 step_mask(step_size - 1);
  wcout << wformat(L"Steps per iteration: %1%") % step_bits << endl;
  wcout << L"Calculating the c and d tables" << endl;
  vector<u8> c;
  u64 c_high(0);
  vector<u32> d;
  u64 d_high(0);
  u32 k(step_bits);
  for (u32 i(0); i < step_size; ++i) {
    u32 odd(0);
    u32 n(i);
    for (u32 j(0); j < k; ++j) {
      if (n & 1) {
        n = (n << 1) + n + 1; // 3 * n + 1;
        n >>= 1; // n / 2
        ++odd;
      }
      else {
        n >>= 1;
      }
    }
    c.push_back(odd);
    d.push_back(n);
    if (odd > c_high) {
      c_high = odd;
    }
    if (n > d_high) {
      d_high = n;
    }
  }
  wcout << wformat(L"c: %1% entries, %2% bytes, %3% high value, 2^%4% high bit")
               % static_cast<u32>(c.size())
               % (static_cast<u32>(c.size()) * sizeof(u8)) % c_high
               % high_bit(c_high)
        << endl;
  wcout << wformat(L"d: %1% entries, %2% bytes, %3% high value, 2^%4% high bit")
               % static_cast<u32>(d.size())
               % (static_cast<u32>(d.size()) * sizeof(u32)) % d_high
               % high_bit(d_high)
        << endl;

  // Calculate 3 to the power of 0 to the highest value stored in the c table.
  //   Find the highest c.
  u32 c_high_power(0);
  for (u32 j(0); j < step_size; ++j) {
    if (c[j] > c_high_power) {
      c_high_power = c[j];
    }
  }
  //   Calculate.
  wcout << wformat(L"Calculating 3 to the power of 0 to %d") % c_high_power
        << endl;
  vector<u32> exp3;
  u64 power_high(0);
  u32 power(1);
  for (u32 j(0); j <= c_high_power; ++j) {
    exp3.push_back(power);
    power *= 3;
    if (power > power_high) {
      power_high = power;
    }
  }
  wcout << wformat(
               L"exp3: %1% entries, %2% bytes, %3% high value, 2^%4% high bit")
               % static_cast<u32>(exp3.size())
               % (static_cast<u32>(exp3.size()) * sizeof(u32)) % power_high
               % high_bit(power_high)
        << endl;

  // Optimization: Early Break if we determine that this N can't possibly beat
  // an existing Delay Record. This optimization is based on the fact that the
  // Delay between two known Delay Records can not be higher than the lower of
  // the two Delay Records. So if any N, regardless of what the starting value
  // was, enters the range between two known Delay Records, we know that the
  // Delay of that N can not become higher than the Delay it has already picked
  // up plus the Delay of the lower of the two records that it's between.
  //
  // For instance, there is a Delay Record at N=77031 D=350 and the next one is
  // at N=106239 D=353. So we know that the highest Delay of any number between
  // 77031 and 106238 is 350. So if an N enters the range of 77031 and 106238
  // (including), it can not possibly get a Delay that is higher than the Delay
  // it has already picked up plus 350. So if its current Delay plus 350 is
  // lower than the current highest known Delay Record, further calculation can
  // be aborted.
  struct delay_record
  {
    delay_record()
    {
    }
    delay_record(u64 n, u32 d) : n_(n), d_(d)
    {
    }
    u64 n_;
    u32 d_;
  };

  u64 n_high(0);
  u32 delay_high(0);

  wcout << L"Loading Delay Records" << endl;
  vector<delay_record> delay_records;
  ifstream delay_file(L".\\delay_records.csv", ios::in);
  if (!delay_file.good()) {
    wcout << L"Warning: Couldn't open delay_records.csv. Starting from scratch"
          << endl;
    // We manually add a dummy Delay Record here, instead of checking for the
    // special case of an empty Delay Record table inside the inner loop.
    delay_records.push_back(delay_record(0, 0));
  }
  else {
    typedef vector<string> part_v_t;
    string line;
    part_v_t parts;
    while (getline(delay_file, line)) {
      split(parts, line, is_any_of(L","));
      if (parts.size() != 2) {
        wcout << L"Error: Incorrectly formatted Delay Record file" << endl;
        throw int(0);
      }
      u64 n(lexical_cast<u64>(parts[0]));
      u32 d(lexical_cast<u32>(parts[1]));
      delay_records.push_back(delay_record(n, d));
    }
    delay_file.close();
    wcout << wformat(L"Loaded %d Delay Records") % delay_records.size() << endl;
    // Find highest Delay Record.
    vector<delay_record>::iterator iter(delay_records.begin());
    n_high = iter->n_;
    delay_high = iter->d_;
    wcout << wformat(L"Highest Delay Record: N: %d, Delay: %d") % n_high
                 % delay_high
          << endl;
  }

  // It is only safe to use the sieve on N higher than the sieve size so we
  // abort if N is not high enough. We should run this test if we don't test
  // within the loop.
  if (n_high <= sieve_size) {
    wcout << wformat(L"Error: N must start above the sieve size (%d)")
                 % sieve_size
          << endl;
    throw int(0);
  }

  // If the highest Delay Record is an even N, we increase it by one because we
  // calculate only odd Ns.
  if (!(n_high & 1)) {
    ++n_high;
  }

  // Optimization: We precalculate the "tail" which is the Delay for all N
  // smaller than a given threshold. So that if N becomes smaller than the
  // threshold, we can look up the Delay instead of calculating it.
  //
  // In an implementation without the multiple steps optimization, this
  // optimization is rendered pretty much worthless by the optimization below
  // that breaks early on hopeless Ns but we need a preculculated tail to
  // compensate for a weakness in the multiple steps optimization.
  //
  // The multiple steps optimization gives an inaccurate Delay if the steps
  // allow N to go all the way down to 1. Since N can fall by 1 bit per step
  // (when N/2 keeps giving an even result), we need a tail that is the same
  // size or wider than the multiple step table widths. For instance, if we're
  // doing 10 steps at a time, we need a tail that is 2^10 = 1024 entries wide.
  const u32 tail_size(1LL << tail_bits);
  wcout << L"Calculating tail" << endl;
  vector<u16> tail;
  u64 tail_high(0);
  for (u32 i(0); i < tail_size; ++i) {
    u32 delay(D(i));
    tail.push_back(delay);
    if (delay > tail_high) {
      tail_high = delay;
    }
  }
  wcout << wformat(
               L"Tail: %1% entries, %2% bits, %3% bytes, %4% high value, 2^%5% "
               L"high bit)")
               % static_cast<u32>(tail.size()) % tail_bits
               % (static_cast<u32>(tail.size()) * sizeof(u16)) % tail_high
               % high_bit(tail_high)
        << endl;

  // Optimization: We only need to calculate odd Ns. Whenever we find an odd
  // Delay Record for N, we know that the delay for 2N = D + 1. So when we find
  // the next Delay Record, we just check if the new N is larger than the old 2N
  // and if it is, the record falls down to the delay of the old 2N.
  u64 two_n(n_high << 1LL);
  u32 two_delay(delay_high + 1);

  // Counters for benchmark.
  u64 stat_vloop_cnt(0);
  u64 stat_early_break(0);
  u64 stat_sieved(0);
  u64 stat_checked_n(0);
  u64 stat_before_sieve(0);
  u64 stat_vloop_cnt_total(0);
  u64 stat_early_break_total(0);
  u64 stat_sieved_total(0);
  u64 stat_checked_n_total(0);
  u64 stat_before_sieve_total(0);

  // Start timers for benchmark.
  timer calc_timer;
  timer calc_timer_total;

  // Calculate delays for Ns starting from the highest known Delay Record to
  // 2^31, which is how far we can safely calculate without a 64 bit overflow.
  // See http://www.ericr.nl/wondrous/pathrecs.html for more on this.
  //
  // We print new Delay Records as they are found.
  wcout << L"Starting calculation\n" << endl;

  // Optimization: We skip all even N (see above).
  for (u64 n(n_high); n < (1ULL << 32); n += 2) {
    // Optimization: We skip all odd N of the form 3k+2 because they can never
    // be a delay record. This is because if N is odd it can not be 6k+2 so it
    // must be 6k+5. This one comes down from 12k+10. This one can be made from
    // 4k+3, which is odd. Since 4k+3 is definitely smaller than 6k+5, and since
    // it's delay is two more, 6k+5 can never be a Delay Record. Therefore no
    // odd N of the form 3k+2 can be a Delay Record.
    //
    // If we were searching for Class Records, we would also be able to skip all
    // N on the form of 3k+2 but we would have to compare any Class Record we
    // found for class R to the record for class R+2 (call it Q). If Q is odd
    // and 3(Q-1)/2+2 is less than the N we found, the Class Record would
    // default to 3(Q-1)/2+2.
    //
    // N is of the form 3k+2 if N is congruent 2 mod 3 (n % 3 == 2).
    if (n % 3 == 2) {
      continue;
    }

    // Keep track of how many N reach the sieve.
    ++stat_before_sieve;

    // Apply the sieve. The sieve can only be used on N that is higher than the
    // sieve size. When we know that we will be working on big Ns, this test is
    // not necessary.
    if (sieve_size && n > sieve_size) {
      u64 half_n(n >> 1);
      u64 bit_idx(half_n & 0x3f);
      u64 word_idx((half_n >> 6) & sieve_mask);
      u64 pick_bit(1ULL << bit_idx);
      if (!(sieve_data[word_idx] & pick_bit)) {
        ++stat_sieved;
        continue;
      }
    }

    // This N survived the filtering. Count it as checked.
    ++stat_checked_n;

    // Fast calculation of D(N)
    u64 n_tmp(n);
    u32 delay(0);
    bool hopeless(false);
    while (n_tmp >= tail_size) {
      // Optimization: Run multiple steps (step_size steps) per iteration.
      u64 b(n_tmp & step_mask);
      u64 a(n_tmp >> step_bits);
      n_tmp = a * exp3[c[b]] + d[b];
      delay += step_bits + c[b];

      // Stats for benchmark.
      stat_vloop_cnt += 1;

      // If N is higher than or equal to N of the highest known Delay Record,
      // we're in uncharted territory and must keep calculating.
      if (n_tmp >= n_high) {
        continue;
      }

      // Early Break optimization. See description above.
      vector<delay_record>::iterator iter(delay_records.begin() + 1);
      while (iter->n_ > n_tmp) {
        ++iter;
      }
      if (delay + iter->d_ <= delay_high) {
        hopeless = true;
        ++stat_early_break;
        break;
      }
    }

    // If the Early Break optimization has flagged this N as hopeless, we skip
    // further processing of this N.
    if (hopeless) {
      continue;
    }

    // Add the preculculated tail for the tail optimization.
    delay += tail[n_tmp];

    // This N has survived the filtering optimizations and the Early Break
    // optimization so it could be a new Delay Record. We check that now.
    if (delay > delay_high) {
      // We have found a new Delay Record.

      // Apply the odd N optimization. If the Delay Record we have found has an
      // N that is more than double of the previous N, we get a Delay Record at
      // the old 2N.
      if (n > two_n) {
        // Add the new Delay Record to the list of known Delay Records used by
        // the Early Break optimization.
        delay_records.insert(
            delay_records.begin(), delay_record(two_n, two_delay));
        // Display the new record.
        wcout << wformat(L"N: %d\tDelay: %d (double)") % two_n % two_delay
              << endl;
      }

      // If the Delay Record we have found has both a Delay and an N that is
      // higher than the two_n record, or if it's just lower than the two_n
      // record, we have another Delay Record.
      if ((n > two_n && delay > two_delay) || n < two_n) {
        // Add the new Delay Record to the list of known Delay Records used by
        // the
        // Early Break optimization.
        delay_records.insert(delay_records.begin(), delay_record(n, delay));
        // Display the new record.
        wcout << wformat(L"N: %d\tDelay: %d") % n % delay << endl;
      }

      // Remember two_n and two_delay for the odd N optimization.
      two_n = n << 1;
      two_delay = delay + 1;

      // Remember this N and this Delay for the Early Break optimization.
      n_high = n;
      delay_high = delay;

      // Benchmark.
      if (benchmark) {
        // Maintain totals
        stat_vloop_cnt_total += stat_vloop_cnt;
        stat_early_break_total += stat_early_break;
        stat_sieved_total += stat_sieved;
        stat_checked_n_total += stat_checked_n;
        stat_before_sieve_total += stat_before_sieve;

        // Display benchmark.
        wcout << L"Benchmark (Delay Record / Total):" << endl;
        wcout << wformat(L"Time: %.2fs / %.2fs") % calc_timer.elapsed()
                     % calc_timer_total.elapsed()
              << endl;
        wcout << wformat(L"Checked: %.2f%% / %.2f%%")
                     % (static_cast<double>(stat_checked_n)
                        / static_cast<double>(n) * 100)
                     % (static_cast<double>(stat_checked_n_total)
                        / static_cast<double>(n) * 100)
              << endl;
        wcout << wformat(L"Early Breaks: %.2f%% / %.2f%%")
                     % (static_cast<double>(stat_early_break)
                        / static_cast<double>(stat_checked_n) * 100)
                     % (static_cast<double>(stat_early_break_total)
                        / static_cast<double>(stat_checked_n_total) * 100)
              << endl;
        wcout << wformat(L"Sieved: %.2f%% / %.2f%%")
                     % (static_cast<double>(stat_sieved)
                        / static_cast<double>(stat_before_sieve) * 100)
                     % (static_cast<double>(stat_sieved_total)
                        / static_cast<double>(stat_before_sieve_total) * 100)
              << endl;
        wcout << wformat(L"Vloops per N: %.2f / %.2f")
                     % (static_cast<double>(stat_vloop_cnt)
                        / static_cast<double>(stat_checked_n))
                     % (static_cast<double>(stat_vloop_cnt_total)
                        / static_cast<double>(stat_checked_n_total))
              << endl;
        wcout << endl;
        // Reset counters.
        stat_vloop_cnt = 0;
        stat_early_break = 0;
        stat_sieved = 0;
        stat_checked_n = 0;
        stat_before_sieve = 0;
        calc_timer.restart();
      }
    }
  }
}

int wmain(int ac, wchar_t** av)
{
  // switch from C locale to user's locale
  locale::global(locale("")); // will be used for all new streams
  wcout.imbue(locale("")); // this stream was already created, so must imbue

  wcout << L"64 Bit Collatz Delay Record Calculator Reference Implementation"
        << endl;
  wcout << L"Submitted to the pulic domain. Roger Dahl, 2008" << endl;
  wcout << L"Thanks to Eric Roosendaal for describing the optimizations"
        << endl;

  // step_bits determines how many steps are taken of the Collatz calculation
  // per iteration and the sizes of the related tables. 12 bit tables are
  // fastest on Intel Core 2 because they fit in the 32K L1 data cache.
  int step_bits;

  // tail_bits determines how many entries are in the tail. The tail must be
  // at least as wide as the step tables.
  int tail_bits;

  // Display benchmark data.
  bool benchmark;

  // Handle program options.
  try {
    po::options_description desc("Options");
    po::variables_map vm;
    desc.add_options()("help,h", "Produce help message")(
        "step_bits,b", po::value<int>(&step_bits)->default_value(12),
        "Width of step table (number of steps per iteration)")(
        "tail_bits,t", po::value<int>(&tail_bits)->default_value(12),
        "Width of tail table")(
        "benchmark,e", po::bool_switch(&benchmark)->default_value(false),
        "Display benchmark data for each Delay Record");

    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }
  } catch (std::exception& e) {
    wcout << L"Error: Couldn't parse command line arguments: " << e.what()
          << endl;
    return 1;
  }

  // Check for limits
  if (step_bits < 1) {
    wcout << L"Error: step_bits must be 1 or more" << endl;
    return 1;
  }

  if (step_bits > 19) {
    wcout << L"Error: step_bits must be 19 or less because higher numbers "
             L"overflow 32 bit words"
          << endl;
    return 1;
  }

  // tail_size must be >= step_size.
  if (tail_bits < step_bits) {
    wcout << L"Error: tail_bits must be equal to or higher than step_bits"
          << endl;
    return 1;
  }

  try {
    collatz_ref(step_bits, tail_bits, benchmark);
  } catch (...) {
    wcout << L"Error: Unhandled exception" << endl;
    return 1;
  }

  return 0;
}