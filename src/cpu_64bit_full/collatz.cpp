#include "core.h"
#include "stdafx.h"

using namespace std;
using namespace boost;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

// Ctrl-C and Ctrl-Break flags.
bool ctrlc_(false);
bool ctrlbrk_(false);

// This handler gets called when ctrl-c is pressed. The handler gets executed in
// a separate thread.
void ctrlc_handler(int sig)
{
  ctrlc_ = true;
}

void ctrlbrk_handler(int sig)
{
  ctrlbrk_ = true;
}

// Find the highest bit set in a word.
u32 high_bit(u64 w)
{
  u32 ret(1);
  while (w >>= 1) {
    ++ret;
  }
  return ret + 1;
}

// Save N Base and Delay.
void save_n_base(u64 n_base, u64 delay)
{
  ofstream n_base_file(L".\\n_base.txt");
  if (!n_base_file.good()) {
    wcout << L"Error: Couldn't write to n_base.txt" << endl;
    throw int(0);
  }
  else {
    n_base_file << n_base << endl;
    n_base_file << delay << endl;
    n_base_file.close();
  }
}

// Save Delay Record.
void save_delay_record(u64 n, u64 delay, bool dbl)
{
  wcout << wformat(L"N: %1%\tDelay: %2% %3%") % n % delay
               % (dbl ? L"(double)" : L"")
        << endl;
  ofstream delay_record_file(L".\\delay_records.txt", ios::out | ios::app);
  if (!delay_record_file.good()) {
    wcout << L"Error: Couldn't write to delay_records.txt" << endl;
    throw int(0);
  }
  else {
    delay_record_file << n << "\t" << delay << endl;
    delay_record_file.close();
  }
}

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

// Collatz Delay Record Calculator.
void collatz(u32 step_bits, u32 tail_bits)
{
  // Load sieve.
  u64 sieve_bytes(0);
  u64 sieve_words(0);
  u64 sieve_size(0);
  vector<u32> sieve_data;
  wcout << L"Loading sieve" << endl;
  ifstream sieve_file(L".\\sieve.dat", ios::binary | ios::in);
  if (!sieve_file.good()) {
    wcout << L"Error: Couldn't open sieve.dat" << endl;
    throw int(0);
  }
  sieve_file.seekg(0, ios::end);
  sieve_bytes = static_cast<u32>(sieve_file.tellg());
  sieve_words = sieve_bytes / sizeof(u32);
  sieve_size = sieve_words * sizeof(u32) * 8;
  sieve_data.resize(sieve_words);
  sieve_file.seekg(0, ios::beg);
  sieve_file.read(reinterpret_cast<char*>(&sieve_data[0]), sieve_bytes);
  sieve_file.close();
  wcout << wformat(L"Sieve: %1% entries, %2% bytes") % sieve_size % sieve_bytes
        << endl;

  // Instead of storing a block of sieve bits where a value of true at a given
  // index tells us to calculate the base plus that index, we store a block of
  // 32 bit indexes to the true values instead. These indexes tell us directly
  // which numbers we need to calculcate and allow us to send a "dense" block of
  // indexes to calculate to the GPU kernel.
  //
  // In addition, all numbers on the form of 3k+2 (one number out of three) can
  // be skipped. We calculate three versions of the table. Each table has a
  // different set of every three numbers removed. At calculation time, we pick
  // one table based on what the Base N for that calculation is.
  wcout << L"Calculating sieve index" << endl;
  vector<vector<u32> > sieve_idx_v;
  for (u64 j(0); j < 3; ++j) {
    u64 sieve_high(0);
    vector<u32> sieve_idx;
    u32 pick_bit(1);
    u32 word_idx(0);
    for (u64 i(0); i < sieve_size; ++i) {
      if (sieve_data[word_idx] & pick_bit) {
        // The sieve contains values only for odd numbers.
        u64 sieve((i << 1) + 1);
        // Filter out values where (n_base + sieve) % 3 == 2.
        if ((j + sieve) % 3 != 2) {
          sieve_idx.push_back(static_cast<u32>(sieve));
          if (sieve > sieve_high) {
            sieve_high = sieve;
          }
        }
      }
      pick_bit <<= 1;
      if (!pick_bit) {
        pick_bit = 1;
        ++word_idx;
      }
    }
    sieve_idx_v.push_back(sieve_idx);
    wcout << wformat(
                 L"Sieve and 3k+2 filter %1%: %2% entries, %3% bytes, %4%%% "
                 L"sieved, %5% high value, 2^%6% high bit")
                 % j % static_cast<u32>(sieve_idx.size())
                 % (static_cast<u32>(sieve_idx.size()) * sizeof(u32))
                 % (100.0 - (static_cast<float>(sieve_idx.size())
                             / static_cast<float>(sieve_size) * 100.0))
                 % sieve_high % high_bit(sieve_high)
          << endl;
  }

  // Calculate step tables.
  const u32 step_size(1LL << step_bits);
  wcout << wformat(L"Steps per iteration: %1%") % step_bits << endl;
  wcout << L"Calculating the c and d tables" << endl;
  vector<u32> c;
  u64 c_high(0);
  vector<u32> d;
  u64 d_high(0);
  for (u64 i(0); i < step_size; ++i) {
    u64 odd(0);
    u64 n(i);
    for (u64 j(0); j < step_bits; ++j) {
      if (n & 1) {
        n = (n << 1) + n + 1; // 3 * n + 1;
        n >>= 1; // n / 2
        ++odd;
      }
      else {
        n >>= 1;
      }
    }
    c.push_back(static_cast<u32>(odd));
    d.push_back(static_cast<u32>(n));
    if (odd > c_high) {
      c_high = odd;
    }
    if (n > d_high) {
      d_high = n;
    }
  }
  wcout << wformat(L"c: %1% entries, %2% bytes, %3% high value, 2^%4% high bit")
               % static_cast<u32>(c.size())
               % (static_cast<u32>(c.size()) * sizeof(u32)) % c_high
               % high_bit(c_high)
        << endl;
  wcout << wformat(L"d: %1% entries, %2% bytes, %3% high value, 2^%4% high bit")
               % static_cast<u32>(d.size())
               % (static_cast<u32>(d.size()) * sizeof(u32)) % d_high
               % high_bit(d_high)
        << endl;

  // The CPU, as opposed to the GPU, is extremely sensitive to the size of the
  // step tables that are used in the inner loop. So, while we simply
  // interleaved 3 32 bit values on the GPU, we crunch the tables together to 1
  // 32 bit big table and one 32 bit small table for the CPU.

  // Check that c and d fit in same 32 bit table.
  if (high_bit(c_high) + high_bit(d_high) > 32) {
    wcout << wformat(L"c high bit + d high bit must be 32 or lower") << endl;
    throw int(0);
  }

  // Make combined c and d table.
  wcout << wformat(L"Creating the combined c and d table") << endl;
  vector<u32> step_table_c_d(c.size());
  u64 c_d_high(0);
  for (u32 i(0); i < c.size(); ++i) {
    u64 c_d((c[i] << high_bit(d_high)) | d[i]);
    step_table_c_d[i] = static_cast<u32>(c_d);
    if (c_d > c_d_high) {
      c_d_high = c_d;
    }
    // wcout << c [i] << "\t" << d [i] << "\t" << ((c [i] << high_bit (d_high))
    // | d [i]) << endl;
  }
  wcout << wformat(
               L"c_d: %1% entries, %2% bytes, %3% high value, 2^%4% high bit")
               % static_cast<u32>(step_table_c_d.size())
               % (static_cast<u32>(step_table_c_d.size()) * sizeof(u32))
               % c_d_high % high_bit(c_d_high)
        << endl;

  // Make exp3 table.
  wcout << wformat(
               L"Calculating the exp3 table (3 to the power of c for each "
               L"value of c)")
        << endl;
  vector<u32> step_table_exp3(step_bits + 1);
  u64 power_high(0);
  for (u32 i(0); i < step_bits + 1; ++i) {
    u64 power(_Pow_int(3, i));
    step_table_exp3[i] = static_cast<u32>(power);
    if (power > power_high) {
      power_high = power;
    }
    // wcout << step_table_exp3 [i] << endl;
  }
  wcout << wformat(
               L"exp3: %1% entries, %2% bytes, %3% high value, 2^%4% high bit")
               % static_cast<u32>(step_table_exp3.size())
               % (static_cast<u32>(step_table_exp3.size()) * sizeof(u32))
               % power_high % high_bit(power_high)
        << endl;

  // Calculate tail.
  wcout << L"Calculating tail" << endl;
  vector<u32> tail;
  u64 tail_high(0);
  for (u64 i(0); i < (1ULL << tail_bits); ++i) {
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
               % (static_cast<u32>(tail.size()) * sizeof(u32)) % tail_high
               % high_bit(tail_high)
        << endl;

  // Find where to start calculating.
  u64 n_base(0);
  u64 delay_high(0);
  wcout << L"Loading N Base" << endl;
  ifstream n_base_file(L".\\n_base.txt", ios::in);
  if (!n_base_file.good()) {
    // Did not find file so will start calculating at beginning.
    //
    // We only have a small buffer for potential Delay Records, so we start the
    // calculation at a point where the buffer is unlikely to overflow.
    wcout << L"Warning: Couldn't open n_base.txt. Starting calculation from "
             L"scratch"
          << endl;
    n_base = sieve_size << (1 + 5);
    delay_high = 1000;
  }
  else {
    // Found the file. Use the first line as the current n_base and second line
    // as current delay_high.
    string line;
    getline(n_base_file, line);
    n_base = lexical_cast<u64>(line);
    getline(n_base_file, line);
    delay_high = lexical_cast<u32>(line);
    n_base_file.close();
    // Make sure we're starting at a valid n_base. If the n_base is not valid,
    // it's set to the closest valid n_base that is lower than the one provided.
    u64 mask(~((sieve_size << 1) - 1));
    n_base &= mask;
  }
  wcout << wformat(L"Starting calculation at N Base: %1%, Delay: %2%") % n_base
               % delay_high
        << endl;

  // It is only safe to use the sieve on N higher than the sieve size.
  if (n_base < sieve_size << 1) {
    wcout << L"Error: N Base must be equal to or higher than sieve size"
          << endl;
    throw int(0);
  }

  // We only need to calculate odd Ns as long as we check for missed records
  // at old N * 2.
  u64 two_n(n_base << 1);
  u64 two_delay(delay_high + 1);

  // Start timers for saving of backups and benchmark.
  timer save_timer;
  timer benchmark_timer;

  // Keep copy of starting N Base for benchmark.
  u64 n_base_start(n_base);

  // Total delay for benchmarking
  u64 total_loops_0(0);
  u64 total_loops_1(0);

  // Calculate delays for Ns starting from the highest known Delay Record and
  // going until Ctrl-C is pressed. Print and save new Delay Records as they are
  // found.
  wcout << L"Starting calculation\n" << endl;

  for (; !ctrlc_; n_base += sieve_size << 1) {
    // Pause on Ctrl-Break.
    if (ctrlbrk_) {
      ctrlbrk_ = false;
      wcout << wformat(L"Ctrl-Break pressed. Sleeping for 2 hours...") << endl;
      thread::sleep(get_system_time() + posix_time::hours(2));
    }

    // Print benchmark and save backup of current n_base and high_delay at
    // regular intervals.
    if (save_timer.elapsed() > 1.0 * 60) {
      save_n_base(n_base, delay_high);
      // Print benchmark.
      wcout << wformat(
                   L"N Base: %1%  Delay: %2%  Speed: %3% N/s  Loops: %4$08x "
                   L"%5$08x")
                   % n_base % delay_high
                   % static_cast<u64>(
                         static_cast<float>(n_base - n_base_start)
                         / benchmark_timer.elapsed())
                   % total_loops_1 % total_loops_0
            << endl;
      total_loops_0 = 0;
      total_loops_1 = 0;
      save_timer.restart();
    }

    // Pick the sieve that filters out 3k+2 values for this n_base.
    u32 sieve_mod(static_cast<u32>(n_base % 3));

    bool odd_n(false);

    while (true) {
      u64 n(n_base);
      u64 delay(delay_high);

      // Attempted to implement this with OpenMP, but that didn't work. See
      // collatz_cpu_unfinished_openmp.cpp for details.

      bool new_record(
          collatz_calc(
              &n, &delay,

              &sieve_idx_v[sieve_mod][0], sieve_idx_v[sieve_mod].size(),

              &tail[0], tail.size(),

              step_bits, high_bit(d_high),
              reinterpret_cast<u32*>(&step_table_c_d[0]),
              reinterpret_cast<u32*>(&step_table_exp3[0]),

              &total_loops_0, &total_loops_1));

      // Odd N optimization.
      if ((!odd_n) && n > two_n) {
        save_delay_record(two_n, two_delay, true);
        odd_n = true;
      }

      // Remember two_n and two_delay for the odd N optimization.
      if (odd_n || new_record) {
        two_n = n << 1;
        two_delay = delay + 1;
      }

      if (new_record) {
        save_delay_record(n, delay, false);
        delay_high = delay;
        continue;
      }
      break;
    }
  }

  // Store n_base so that we can start at the same spot in the next run.
  save_n_base(n_base, delay_high);
}

int main(int ac, char** av)
{
  // Set current directory to where exe is and store it (current directory is a
  // global variable that should never be changed)
  char exe[MAX_PATH];
  GetModuleFileNameA(NULL, exe, MAX_PATH);
  fs::path exe_path(exe);
  SetCurrentDirectoryA(
      exe_path.branch_path().native_directory_string().c_str());
  // Initial_path is a global variable that can be changed only once.
  // boost::filesystem uses it for creating absolute paths from relative paths
  // (fs::system_complete)
  fs::initial_path();

  // switch from C locale to user's locale
  locale::global(locale("")); // Will be used for all new streams, such as temp
  // streams created by wformat().
  wcout.imbue(locale("")); // This stream was already created, so must imbue.

  // Set the standard precision for output of floats.
  wcout.precision(4);

  // Brag.
  wcout << L"128 bit Collatz Delay Record Calculator (C)2008 Roger Dahl"
        << endl;
  wcout << L"Thanks to Eric Roosendaal for describing the high-level "
           L"optimizations\n"
        << endl;

  // step_bits determines how many steps are taken per iteration in the inner
  // calculation loop and also the sizes of the related tables. The highest
  // step_bits that is possible with 32 bit tables is 15 and that is also the
  // fastest.
  u32 step_bits;

  // tail_bits determines how many entries are in the tail. The tail must be
  // at least as wide as the step tables.
  u32 tail_bits;

  // Handle program options.
  po::options_description desc("Options");
  po::variables_map vm;
  desc.add_options()("help,h", "Produce help message")(
      "step_bits,b", po::value<u32>(&step_bits)->default_value(15),
      "Width of step table (number of steps per iteration)")(
      "tail_bits,t", po::value<u32>(&tail_bits)->default_value(15),
      "Width of tail table");
  try {
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << desc << endl;
      return 0;
    }
  } catch (std::exception& e) {
    wcout << L"Error: Couldn't parse command line arguments: " << e.what()
          << "\n"
          << endl;
    cout << desc << endl;
    return 1;
  }

  // Check for limits

  if (step_bits < 1) {
    wcout << L"Error: step_bits must be 1 or more" << endl;
    return 1;
  }

  if (step_bits > 15) {
    wcout << L"Error: step_bits must be 15 or less because higher numbers "
             L"cause 32 bit overflow"
          << endl;
    return 1;
  }

  // tail_size must be >= step_size.
  if (tail_bits < step_bits) {
    wcout << L"Error: tail_bits must be equal to or higher than step_bits"
          << endl;
    return 1;
  }

  // Set up signal handlers.
  // Ctrl-C
  signal(SIGINT, ctrlc_handler);
  // Ctrl-Break
  signal(SIGBREAK, ctrlbrk_handler);

  // Run the calculation.
  try {
    collatz(step_bits, tail_bits);
  } catch (...) {
    wcout << L"Error: Unhandled exception" << endl;
    // Error exit.
    return 1;
  }

  // Successful exit.
  return 0;
}
