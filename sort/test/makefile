OPTS = -O3 -I. -I.. --std c++20  -arch=sm_86

all: test test_hyb test_ctrl test_hybcount

test: test.cu
	nvcc $(OPTS)   -o $@  $<

test_hyb: test_hyb.cu
	nvcc $(OPTS)   -o $@  $<

test_ctrl: test_ctrl.cu
	nvcc $(OPTS)   -o $@  $<

test_hybcount: test_hybcount.cu
	nvcc $(OPTS)   -o $@  $<

validate: clean test test_ctrl test_hyb test_hybcount
	./test_ctrl > rng_test_ctrl.txt
	@ export RES="pass" ; diff rng_test_ctrl.txt  rngs_sorted.txt ||  export RES="fail" ; echo "Test: $$RES"
	./test > rng_test.txt
	@ export RES="pass" ; diff rng_test.txt  rngs_sorted.txt ||  export RES="fail" ; echo "Test: $$RES"
	./test_hyb > rng_test_hyb.txt
	@ export RES="pass" ; diff rng_test_hyb.txt  rngs_sorted.txt ||  export RES="fail" ; echo "Test: $$RES"
	./test_hybcount > rng_test_hybcount.txt
	@ export RES="pass" ; diff rng_test_hybcount.txt  rngs_sorted.txt ||  export RES="fail" ; echo "Test: $$RES"

clean:
	rm -f test test_hyb test_ctrl rng_test_ctrl.txt rng_test.txt rng_test_hyb.txt rng_test_hybcount.txt test_hybcount
