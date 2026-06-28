#pragma once
// Shared function-pointer type for all elementwise binary custom ops.
// Signature: (input_a, input_b, output, element_count)

namespace ds4 {
typedef void (*binop_fn)(const float * a, const float * b, float * n, int count);
} // namespace ds4
