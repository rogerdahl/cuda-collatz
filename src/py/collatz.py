n1 = 0
dmax = 0
while 1:
	n1 += 1
	n = n1
	d = 0
	while n > 1:
		d += 1
		if n & 1 > 0:
			n = 3 * n + 1
		else:
			n >>= 1;

	if d > dmax:
		print n1, '\t', d
		dmax = d