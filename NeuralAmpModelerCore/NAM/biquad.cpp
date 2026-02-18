#include "biquad.h"

namespace nam
{

// ---- Biquad ----

Biquad::Biquad()
: _b0(1.0f)
, _b1(0.0f)
, _b2(0.0f)
, _a1(0.0f)
, _a2(0.0f)
, _s1(0.0f)
, _s2(0.0f)
{
}

void Biquad::SetCoefficients(float b0, float b1, float b2, float a1, float a2)
{
  _b0 = b0;
  _b1 = b1;
  _b2 = b2;
  _a1 = a1;
  _a2 = a2;
}

void Biquad::set_weights_(std::vector<float>::iterator& weights)
{
  _b0 = *weights++;
  _b1 = *weights++;
  _b2 = *weights++;
  _a1 = *weights++;
  _a2 = *weights++;
}

void Biquad::Process(float* data, int num_frames)
{
  for (int n = 0; n < num_frames; ++n)
  {
    data[n] = ProcessSample(data[n]);
  }
}

float Biquad::ProcessSample(float input)
{
  float output = _b0 * input + _s1;
  _s1 = _b1 * input - _a1 * output + _s2;
  _s2 = _b2 * input - _a2 * output;
  return output;
}

void Biquad::Reset()
{
  _s1 = 0.0f;
  _s2 = 0.0f;
}

// ---- BiquadCascade ----

BiquadCascade::BiquadCascade(int num_sections)
: _sections(static_cast<size_t>(num_sections))
{
}

void BiquadCascade::set_weights_(std::vector<float>::iterator& weights)
{
  for (auto& section : _sections)
  {
    section.set_weights_(weights);
  }
}

void BiquadCascade::Process(float* data, int num_frames)
{
  for (auto& section : _sections)
  {
    section.Process(data, num_frames);
  }
}

void BiquadCascade::Reset()
{
  for (auto& section : _sections)
  {
    section.Reset();
  }
}

} // namespace nam
