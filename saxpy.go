package main

//go:generate nvcc saxpy.cu --ptx
//go:generate nvcc -c saxpy.cu
//go:generate ar rvs saxpy.a saxpy.o

//#cgo LDFLAGS: -L/opt/cuda/lib64 -lcuda -lcudart saxpy.a
import "C"
import (
	"fmt"
	"github.com/mumax/3/cuda/cu"
	"runtime"
	"unsafe"
)

func main() {
	runtime.LockOSThread()

	cu.Init(0)

	cu.CtxCreate(cu.CTX_BLOCKING_SYNC, 0).SetCurrent()

	mod := cu.ModuleLoad("saxpy.ptx")
	saxpy := mod.GetFunction("saxpy")

	const N = 1 << 20
	const N4 = int64(N * unsafe.Sizeof(float32(0)))

	xs := make([]float32, N)
	ys := make([]float32, N)

	xs_d := cu.MemAlloc(N4)
	ys_d := cu.MemAlloc(N4)

	for i := 0; i < N; i++ {
		xs[i] = 1.0
		ys[i] = 2.0
	}

	cu.MemcpyHtoD(xs_d, unsafe.Pointer(&xs[0]), N4)
	cu.MemcpyHtoD(ys_d, unsafe.Pointer(&ys[0]), N4)

	var n int32 = N
	var a float32 = 2
	args := []unsafe.Pointer{
		unsafe.Pointer(&n),
		unsafe.Pointer(&a),
		unsafe.Pointer(&xs_d),
		unsafe.Pointer(&ys_d),
	}
	cu.LaunchKernel(saxpy, (N+255)/256, 1, 1, 256, 1, 1, 0, 0, args)

	cu.MemcpyDtoH(unsafe.Pointer(&ys[0]), ys_d, N4)

	var maxError float32
	for i := 0; i < N; i++ {
		maxError = maxFloat32(maxError, absFloat32(ys[i]-4.0))
	}
	fmt.Println("Max error:", maxError)
}

func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	} else {
		return b
	}
}

func absFloat32(a float32) float32 {
	if a < 0 {
		return -a
	} else {
		return a
	}
}
