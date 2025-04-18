# Mini AutoGrad

A lightweight C++ library implementing basic linear algebra tools and automatic differentiation.
It is mostly inspired by

- Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., & Lerer, A. (2017). *Automatic differentiation in PyTorch*. In NIPS Autodiff Workshop. [link](https://openreview.net/forum?id=BJJsrmfCZ)

## Requirements

- C++20 compatible compiler (I used g++ myself)
- Standard C++ libraries

## Build Instructions

To build the project:

```bash
make
```

To clean build artifacts:

```bash
make clean
```

## Usage Example

```cpp
#include "tensor.hpp"
#include "functional.hpp"
#include <iostream>

int main() {
    // Create tensors
    Tensor x({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);
    Tensor y({0.1, 0.2, 0.3, 0.4}, {2, 2}, true);
    
    // Perform operations
    Tensor z = add(x, y);
    
    // Compute gradients
    z.backward();
    
    // Print results
    std::cout << "x: " << x.to_string() << std::endl;
    std::cout << "y: " << y.to_string() << std::endl;
    std::cout << "z = x + y: " << z.to_string() << std::endl;
    
    return 0;
}
```

## License

You can do whatever you want with it, but why would you?..