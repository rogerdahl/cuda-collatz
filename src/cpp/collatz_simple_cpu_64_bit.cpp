// Simplest possible CPU implementation in C++

#include "stdafx.h"

typedef unsigned long long u64;
typedef unsigned int u32;

int _tmain(int argc, _TCHAR* argv[])
{
  DWORD time(GetTickCount());

  u32 delay_high(0);
  for (u32 i(0); i < (1 << 26); ++i) {
    u64 n(i);
    u32 delay(0);

    while (n > 1) {
      if (n & 1) {
        n = (n << 1) + n + 1;
        ++delay;
      }
      n >>= 1;
      ++delay;
    }

    if (delay > delay_high) {
      delay_high = delay;
      printf("%d ", delay);
    }
  }

  printf("\n%.1f\n", float(GetTickCount() - time) / float(1000));
  return 0;
}
