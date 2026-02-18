#pragma once
// parametric_amp.h — Parametric amp model for real-time inference
//
// Four-stage cascade: Preamp -> Tone Stack -> Power Amp -> Output
// Each stage uses biquad filters, GRU nonlinearities, and MLP knob controllers.

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "biquad.h"
#include "dsp.h"
#include "gru.h"
#include "json.hpp"

namespace nam
{
namespace parametric
{

/// Small MLP mapping knob values [0,1] to internal DSP parameters
///
/// Architecture: Linear(knob_dim, hidden) -> Tanh -> Linear(hidden, hidden) -> Tanh -> Linear(hidden, output)
class KnobController
{
public:
  KnobController(int knob_dim, int output_dim, int hidden_dim = 16);

  /// Load weights from iterator
  void set_weights_(std::vector<float>::iterator& weights);

  /// Compute DSP parameters from knob values
  /// \param knobs Input knob values (knob_dim elements)
  /// \param output Output DSP parameters (output_dim elements)
  void Process(const float* knobs, float* output);

private:
  int _knob_dim, _output_dim, _hidden_dim;
  Eigen::MatrixXf _w1, _w2, _w3;
  Eigen::VectorXf _b1, _b2, _b3;

  // Scratch buffers
  Eigen::VectorXf _input;
  Eigen::VectorXf _h1, _h2;
};

/// Preamp stage: biquad_pre -> gain -> GRU nonlinearity -> biquad_post
class PreampStage
{
public:
  PreampStage(int num_biquad_pre = 2, int num_biquad_post = 2, int gru_hidden = 1);

  void set_weights_(std::vector<float>::iterator& weights);
  void Process(float* data, int num_frames, float preamp_gain_knob);
  void Reset();

private:
  BiquadCascade _pre_filter;
  BiquadCascade _post_filter;
  GRUCell _nonlinearity;
  KnobController _knob_ctrl; // 1 knob -> 2 outputs (gain, bias)
};

/// Tone stack: 3 biquads whose coefficients come from an MLP controller
class ToneStackStage
{
public:
  ToneStackStage();

  void set_weights_(std::vector<float>::iterator& weights);
  void Process(float* data, int num_frames, const float* tone_knobs); // bass, mid, treble
  void Reset();

private:
  std::array<Biquad, 3> _filters; // low shelf, peak, high shelf
  KnobController _knob_ctrl;      // 3 knobs -> 15 coefficients (5 per filter)
  std::array<float, 15> _coeffs;  // Cached filter coefficients
};

/// Power amp: envelope -> sag -> WH core -> negative feedback
class PowerAmpStage
{
public:
  PowerAmpStage(int gru_hidden = 1, int num_biquad = 2);

  void set_weights_(std::vector<float>::iterator& weights);
  void Process(float* data, int num_frames, const float* pa_knobs); // sag, presence, depth
  void Reset();

private:
  BiquadCascade _pre_filter;
  BiquadCascade _post_filter;
  GRUCell _nonlinearity;

  // Envelope follower state
  float _envelope = 0.0f;
  float _attack_coeff = 0.0f;
  float _release_coeff = 0.0f;
  float _raw_attack = 0.0f;
  float _raw_release = 0.0f;

  KnobController _sag_ctrl;     // 1 knob -> 1 output (sag depth)
  KnobController _feedback_ctrl; // 2 knobs -> 5 coefficients (1 feedback biquad)
  Biquad _feedback_filter;
  float _feedback_amount = 0.0f;
};

/// Output stage: GRU -> biquad -> master volume
class OutputStage
{
public:
  OutputStage(int gru_hidden = 1);

  void set_weights_(std::vector<float>::iterator& weights);
  void Process(float* data, int num_frames, float master_vol_knob);
  void Reset();

private:
  GRUCell _nonlinearity;
  BiquadCascade _filter;
  KnobController _volume_ctrl; // 1 knob -> 1 output (volume)
};

// ---- Main ParametricAmp class ----

static constexpr int NUM_KNOBS = 8;

/// Complete parametric amp model inheriting from DSP base
///
/// Knob indices:
///   0: preamp_gain
///   1: bass
///   2: mid
///   3: treble
///   4: sag
///   5: presence
///   6: depth
///   7: master_volume
class ParametricAmp : public DSP
{
public:
  ParametricAmp(const nlohmann::json& config, std::vector<float>& weights, double expected_sample_rate);
  ~ParametricAmp() override = default;

  /// Process audio through the four-stage chain
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

  /// Set a knob value (0.0 to 1.0)
  void SetKnob(int index, float value);

  /// Get the current value of a knob
  float GetKnob(int index) const;

  /// Get the number of parametric knobs
  int GetNumKnobs() const { return NUM_KNOBS; }

  /// Get the name of a knob by index
  const std::string& GetKnobName(int index) const;

  /// Reset all internal states
  void Reset(const double sampleRate, const int maxBufferSize) override;

protected:
  void SetMaxBufferSize(const int maxBufferSize) override;
  int PrewarmSamples() override { return 48000; } // 1 second

private:
  PreampStage _preamp;
  ToneStackStage _tonestack;
  PowerAmpStage _poweramp;
  OutputStage _output;

  std::array<float, NUM_KNOBS> _knobs;
  std::vector<std::string> _knob_names;

  // Processing buffer
  std::vector<float> _buffer;
};

/// Factory function for the FactoryRegistry
std::unique_ptr<DSP> Factory(const nlohmann::json& config, std::vector<float>& weights, const double expectedSampleRate);

} // namespace parametric
} // namespace nam
