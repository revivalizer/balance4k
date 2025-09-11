#version 430
// CREDIT: Template from https://github.com/yosshin4004/minimal_gl/blob/master/examples/04_sound_output.snd.glsl, says "Copyright (C) 2020 Yosshin(@yosshin4004)"

layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
	#pragma work_around_begin:layout(std430,binding=0)buffer ssbo{vec2 %s[];};layout(local_size_x=1)in;
	vec2 waveOutSamples[];
	#pragma work_around_end
#else
	layout(std430, binding = 0) buffer SoundOutput{ vec2 waveOutSamples[]; };
	layout(local_size_x = 1) in;
#endif


#define NUM_SAMPLES_PER_SEC 48000.
void main(){
	int offset = int(gl_GlobalInvocationID.x) + waveOutPosition;
	float sec = float(offset) / NUM_SAMPLES_PER_SEC;
	waveOutSamples[offset] = sin(vec2(sec * 440 * 6.2831)) * exp(-sec);
}

