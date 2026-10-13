// Loads each custom-op library given on the command line with dlopen and checks that the ORT custom-op entry point
// `RegisterCustomOps` is exported. Needs neither a GPU nor an ORT runtime.
#include <dlfcn.h>

#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s lib.so...\n", argv[0]);
    return 2;
  }
  int failures = 0;
  for (int i = 1; i < argc; ++i) {
    void* handle = dlopen(argv[i], RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      std::fprintf(stderr, "FAIL dlopen %s: %s\n", argv[i], dlerror());
      ++failures;
      continue;
    }
    if (!dlsym(handle, "RegisterCustomOps")) {
      std::fprintf(stderr, "FAIL %s: RegisterCustomOps not exported\n", argv[i]);
      ++failures;
    } else {
      std::printf("OK   %s\n", argv[i]);
    }
    dlclose(handle);
  }
  return failures ? 1 : 0;
}
