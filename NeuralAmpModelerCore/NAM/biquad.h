#pragma once
// biquad.h — Direct Form II Transposed biquad filter for parametric amp
//
// Lightweight IIR filter processing audio sample-by-sample.
// Used for tone stack EQ, pre/post filters in WH stages, and feedback paths.

#include <vector>

namespace nam
{

/// Direct Form II Transposed biquad filter
///
/// Transfer function:
///   H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
///
/// State equations:
///   y[n] = b0*x[n] + s1
///   s1   = b1*x[n] - a1*y[n] + s2
///   s2   = b2*x[n] - a2*y[n]
class Biquad
{
public:
  Biquad();

  /// Set filter coefficients
  /// \param b0, b1, b2 Feed-forward coefficients
  /// \param a1, a2 Feed-back coefficients (a0 = 1 implied)
  void SetCoefficients(float b0, float b1, float b2, float a1, float a2);

  /// Load coefficients from a weight iterator (for .nam file loading)
  void set_weights_(std::vector<float>::iterator& weights);

  /// Process a block of audio in-place
  void Process(float* data, int num_frames);

  /// Process a single sample (for fine-grained control)
  float ProcessSample(float input);

  /// Reset filter state to zero
  void Reset();

private:
  float _b0 = 1.0f, _b1 = 0.0f, _b2 = 0.0f;
  float _a1 = 0.0f, _a2 = 0.0f;
  float _s1 = 0.0f, _s2 = 0.0f; // State variables
};

/// Cascade of N biquad filter sections in series
class BiquadCascade
{
public:
  BiquadCascade() = default;
  explicit BiquadCascade(int num_sections);

  /// Load all section coefficients from weight iterator
  void set_weights_(std::vector<float>::iterator& weights);

  /// Process a block of audio in-place (cascades through all sections)
  void Process(float* data, int num_frames);

  /// Reset all filter states
  void Reset();

  int GetNumSections() const { return static_cast<int>(_sections.size()); }

private:
  std::vector<Biquad> _sections;
};

} // namespace nam
