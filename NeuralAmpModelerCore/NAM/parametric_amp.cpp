#include "parametric_amp.h"
#include "registry.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nam
{
namespace parametric
{

// ---- KnobController ----

KnobController::KnobController(int knob_dim, int output_dim, int hidden_dim)
: _knob_dim(knob_dim)
, _output_dim(output_dim)
, _hidden_dim(hidden_dim)
, _w1(hidden_dim, knob_dim)
, _w2(hidden_dim, hidden_dim)
, _w3(output_dim, hidden_dim)
, _b1(hidden_dim)
, _b2(hidden_dim)
, _b3(output_dim)
, _input(knob_dim)
, _h1(hidden_dim)
, _h2(hidden_dim)
{
  _w1.setZero();
  _w2.setZero();
  _w3.setZero();
  _b1.setZero();
  _b2.setZero();
  _b3.setZero();
}

void KnobController::set_weights_(std::vector<float>::iterator& weights)
{
  // PyTorch nn.Linear stores weight as (out_features, in_features) row-major
  // then bias as (out_features)
  auto load_linear = [&](Eigen::MatrixXf& w, Eigen::VectorXf& b) {
    for (int r = 0; r < w.rows(); ++r)
      for (int c = 0; c < w.cols(); ++c)
        w(r, c) = *weights++;
    for (int i = 0; i < b.size(); ++i)
      b(i) = *weights++;
  };

  load_linear(_w1, _b1);
  load_linear(_w2, _b2);
  load_linear(_w3, _b3);
}

void KnobController::Process(const float* knobs, float* output)
{
  // Map input to Eigen
  for (int i = 0; i < _knob_dim; ++i)
    _input(i) = knobs[i];

  // Layer 1: tanh(W1 * x + b1)
  _h1 = (_w1 * _input + _b1).array().tanh().matrix();

  // Layer 2: tanh(W2 * h1 + b2)
  _h2 = (_w2 * _h1 + _b2).array().tanh().matrix();

  // Layer 3: W3 * h2 + b3 (no activation)
  Eigen::VectorXf out = _w3 * _h2 + _b3;

  for (int i = 0; i < _output_dim; ++i)
    output[i] = out(i);
}

// ---- PreampStage ----

PreampStage::PreampStage(int num_biquad_pre, int num_biquad_post, int gru_hidden)
: _pre_filter(num_biquad_pre)
, _post_filter(num_biquad_post)
, _nonlinearity(1, gru_hidden)
, _knob_ctrl(1, 2, 16)
{
}

void PreampStage::set_weights_(std::vector<float>::iterator& weights)
{
  _pre_filter.set_weights_(weights);
  _post_filter.set_weights_(weights);
  _nonlinearity.set_weights_(weights);
  _knob_ctrl.set_weights_(weights);
}

void PreampStage::Process(float* data, int num_frames, float preamp_gain_knob)
{
  // Get gain parameters from knob controller
  float params[2];
  _knob_ctrl.Process(&preamp_gain_knob, params);
  float gain_mult = std::exp(params[0] * 4.0f - 2.0f);
  float bias = params[1] * 0.1f;

  // Pre-filter
  _pre_filter.Process(data, num_frames);

  // Apply gain + bias, then GRU nonlinearity
  for (int n = 0; n < num_frames; ++n)
  {
    data[n] = data[n] * gain_mult + bias;
    data[n] = _nonlinearity.ProcessSample(data[n]);
  }

  // Post-filter
  _post_filter.Process(data, num_frames);
}

void PreampStage::Reset()
{
  _pre_filter.Reset();
  _post_filter.Reset();
  _nonlinearity.Reset();
}

// ---- ToneStackStage ----

ToneStackStage::ToneStackStage()
: _knob_ctrl(3, 15, 16) // 3 knobs -> 15 coefficients (5 per filter * 3 filters)
{
  _coeffs.fill(0.0f);
}

void ToneStackStage::set_weights_(std::vector<float>::iterator& weights)
{
  _knob_ctrl.set_weights_(weights);
}

void ToneStackStage::Process(float* data, int num_frames, const float* tone_knobs)
{
  // Get filter coefficients from knob controller
  _knob_ctrl.Process(tone_knobs, _coeffs.data());

  // Apply stability enforcement (radius/angle parameterization)
  // and set coefficients for each filter
  for (int f = 0; f < 3; ++f)
  {
    float b0 = _coeffs[f * 5 + 0];
    float b1 = _coeffs[f * 5 + 1];
    float b2 = _coeffs[f * 5 + 2];

    // Stability via sigmoid -> radius/angle
    float raw_radius = _coeffs[f * 5 + 3];
    float raw_angle = _coeffs[f * 5 + 4];
    float radius = 1.0f / (1.0f + std::exp(-raw_radius)) * 0.999f;
    float angle = 1.0f / (1.0f + std::exp(-raw_angle)) * 3.14159265f;
    float a1 = -2.0f * radius * std::cos(angle);
    float a2 = radius * radius;

    _filters[f].SetCoefficients(b0, b1, b2, a1, a2);
  }

  // Apply each filter in cascade
  for (auto& filter : _filters)
  {
    filter.Process(data, num_frames);
  }
}

void ToneStackStage::Reset()
{
  for (auto& f : _filters)
    f.Reset();
}

// ---- PowerAmpStage ----

PowerAmpStage::PowerAmpStage(int gru_hidden, int num_biquad)
: _pre_filter(num_biquad)
, _post_filter(num_biquad)
, _nonlinearity(1, gru_hidden)
, _sag_ctrl(1, 1, 8)
, _feedback_ctrl(2, 5, 16) // presence + depth -> 5 biquad coeffs
{
}

void PowerAmpStage::set_weights_(std::vector<float>::iterator& weights)
{
  _pre_filter.set_weights_(weights);
  _post_filter.set_weights_(weights);
  _nonlinearity.set_weights_(weights);

  // Envelope follower params
  _raw_attack = *weights++;
  _raw_release = *weights++;

  _sag_ctrl.set_weights_(weights);
  _feedback_ctrl.set_weights_(weights);

  _feedback_filter.Reset();
  _feedback_amount = *weights++;
}

void PowerAmpStage::Process(float* data, int num_frames, const float* pa_knobs)
{
  float sag_knob = pa_knobs[0];
  float feedback_knobs[2] = {pa_knobs[1], pa_knobs[2]}; // presence, depth

  // Get sag depth
  float sag_param;
  _sag_ctrl.Process(&sag_knob, &sag_param);
  float sag_depth = 1.0f / (1.0f + std::exp(-sag_param)); // sigmoid

  // Compute envelope follower coefficients
  float attack_ms = 0.1f + 1.0f / (1.0f + std::exp(-_raw_attack)) * 49.9f;
  float release_ms = 10.0f + 1.0f / (1.0f + std::exp(-_raw_release)) * 490.0f;
  _attack_coeff = std::exp(-1.0f / (attack_ms * 48.0f)); // Assuming 48kHz
  _release_coeff = std::exp(-1.0f / (release_ms * 48.0f));

  // Get feedback filter coefficients
  float fb_raw[5];
  _feedback_ctrl.Process(feedback_knobs, fb_raw);

  // Apply stability enforcement
  float fb_radius = 1.0f / (1.0f + std::exp(-fb_raw[3])) * 0.999f;
  float fb_angle = 1.0f / (1.0f + std::exp(-fb_raw[4])) * 3.14159265f;
  _feedback_filter.SetCoefficients(
    fb_raw[0], fb_raw[1], fb_raw[2],
    -2.0f * fb_radius * std::cos(fb_angle),
    fb_radius * fb_radius);

  float fb_mix = 1.0f / (1.0f + std::exp(-_feedback_amount)); // sigmoid

  // Process each sample
  for (int n = 0; n < num_frames; ++n)
  {
    // Envelope tracking
    float level = std::abs(data[n]);
    float coeff = (level > _envelope) ? _attack_coeff : _release_coeff;
    _envelope = coeff * _envelope + (1.0f - coeff) * level;

    // Sag gain
    float sag_gain = 1.0f - sag_depth * _envelope;
    data[n] *= sag_gain;
  }

  // WH core
  _pre_filter.Process(data, num_frames);
  for (int n = 0; n < num_frames; ++n)
  {
    data[n] = _nonlinearity.ProcessSample(data[n]);
  }
  _post_filter.Process(data, num_frames);

  // Negative feedback
  // Note: simplified — feedback applied after full WH rather than within loop
  // This avoids per-sample feedback filter + main chain coupling which is expensive
  for (int n = 0; n < num_frames; ++n)
  {
    float fb = _feedback_filter.ProcessSample(data[n]);
    data[n] -= fb_mix * fb;
  }
}

void PowerAmpStage::Reset()
{
  _pre_filter.Reset();
  _post_filter.Reset();
  _nonlinearity.Reset();
  _feedback_filter.Reset();
  _envelope = 0.0f;
}

// ---- OutputStage ----

OutputStage::OutputStage(int gru_hidden)
: _nonlinearity(1, gru_hidden)
, _filter(1)
, _volume_ctrl(1, 1, 8)
{
}

void OutputStage::set_weights_(std::vector<float>::iterator& weights)
{
  _nonlinearity.set_weights_(weights);
  _filter.set_weights_(weights);
  _volume_ctrl.set_weights_(weights);
}

void OutputStage::Process(float* data, int num_frames, float master_vol_knob)
{
  // GRU nonlinearity
  for (int n = 0; n < num_frames; ++n)
  {
    data[n] = _nonlinearity.ProcessSample(data[n]);
  }

  // Output filter
  _filter.Process(data, num_frames);

  // Master volume
  float vol_param;
  _volume_ctrl.Process(&master_vol_knob, &vol_param);
  float volume = std::exp(vol_param * 4.0f - 4.0f);

  for (int n = 0; n < num_frames; ++n)
  {
    data[n] *= volume;
  }
}

void OutputStage::Reset()
{
  _nonlinearity.Reset();
  _filter.Reset();
}

// ---- ParametricAmp ----

ParametricAmp::ParametricAmp(
  const nlohmann::json& config,
  std::vector<float>& weights,
  double expected_sample_rate)
: DSP(1, 1, expected_sample_rate)
, _preamp(
    config.value("preamp", nlohmann::json{}).value("num_biquad_sections_pre", 2),
    config.value("preamp", nlohmann::json{}).value("num_biquad_sections_post", 2),
    config.value("preamp", nlohmann::json{}).value("gru_hidden_size", 1))
, _tonestack()
, _poweramp(
    config.value("poweramp", nlohmann::json{}).value("gru_hidden_size", 1),
    config.value("poweramp", nlohmann::json{}).value("num_biquad_sections", 2))
, _output(
    config.value("output", nlohmann::json{}).value("gru_hidden_size", 1))
{
  // Initialize knobs to nominal (0.5)
  _knobs.fill(0.5f);

  // Set knob names
  _knob_names = {
    "preamp_gain", "bass", "mid", "treble",
    "sag", "presence", "depth", "master_volume"
  };

  // Load weights into each stage
  auto it = weights.begin();
  _preamp.set_weights_(it);
  _tonestack.set_weights_(it);
  _poweramp.set_weights_(it);
  _output.set_weights_(it);
}

void ParametricAmp::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  // Resize buffer if needed
  if (static_cast<int>(_buffer.size()) < num_frames)
    _buffer.resize(num_frames);

  // Copy input to processing buffer (convert from NAM_SAMPLE to float)
  for (int n = 0; n < num_frames; ++n)
    _buffer[n] = static_cast<float>(input[0][n]);

  // Process through each stage
  _preamp.Process(_buffer.data(), num_frames, _knobs[0]);

  float tone_knobs[3] = {_knobs[1], _knobs[2], _knobs[3]};
  _tonestack.Process(_buffer.data(), num_frames, tone_knobs);

  float pa_knobs[3] = {_knobs[4], _knobs[5], _knobs[6]};
  _poweramp.Process(_buffer.data(), num_frames, pa_knobs);

  _output.Process(_buffer.data(), num_frames, _knobs[7]);

  // Copy to output (convert from float to NAM_SAMPLE)
  for (int n = 0; n < num_frames; ++n)
    output[0][n] = static_cast<NAM_SAMPLE>(_buffer[n]);
}

void ParametricAmp::SetKnob(int index, float value)
{
  if (index >= 0 && index < NUM_KNOBS)
    _knobs[index] = std::clamp(value, 0.0f, 1.0f);
}

float ParametricAmp::GetKnob(int index) const
{
  if (index >= 0 && index < NUM_KNOBS)
    return _knobs[index];
  return 0.0f;
}

const std::string& ParametricAmp::GetKnobName(int index) const
{
  static const std::string empty;
  if (index >= 0 && index < NUM_KNOBS)
    return _knob_names[index];
  return empty;
}

void ParametricAmp::Reset(const double sampleRate, const int maxBufferSize)
{
  DSP::Reset(sampleRate, maxBufferSize);
  _preamp.Reset();
  _tonestack.Reset();
  _poweramp.Reset();
  _output.Reset();
}

void ParametricAmp::SetMaxBufferSize(const int maxBufferSize)
{
  DSP::SetMaxBufferSize(maxBufferSize);
  _buffer.resize(maxBufferSize);
}

// ---- Factory ----

std::unique_ptr<DSP> Factory(
  const nlohmann::json& config,
  std::vector<float>& weights,
  const double expectedSampleRate)
{
  return std::make_unique<ParametricAmp>(config, weights, expectedSampleRate);
}

// Register with the factory
static factory::Helper _register_ParametricAmp("ParametricAmp", Factory);

} // namespace parametric
} // namespace nam
