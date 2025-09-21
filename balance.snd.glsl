#version 430
// CREDIT: Template from https://github.com/yosshin4004/minimal_gl/blob/master/examples/04_sound_output.snd.glsl, says "Copyright (C) 2020 Yosshin(@yosshin4004)"

// CREDIT: FROM brainfiller/Ob5vr (https://github.com/0b5vr/brainfiller/blob/main/brainfiller.snd.glsl)
#define repeat(i, n) for(int i=0; i<(n); i++)
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
	#pragma work_around_begin:layout(std430,binding=0)buffer ssbo{vec2 %s[];};layout(local_size_x=1)in;
	vec2 waveOutSamples[];
	#pragma work_around_end
#else
	layout(std430, binding = 0) buffer SoundOutput{ vec2 waveOutSamples[]; };
	layout(local_size_x = 1) in;
#endif

const float BPM = 170;
const float BPS = BPM / 60.f;

const float SAMPLES_PER_SEC = 48000.; // 48000 in minimal_gl

const int SAMPLES_PER_STEP = int( SAMPLES_PER_SEC / BPS / 4.0 );
const int SAMPLES_PER_BEAT = 4 * SAMPLES_PER_STEP;

const float T2B = BPS;
const float B2T = 1.0 / BPS;
//const float S2T = 0.25 * B2T;

// vec2 kick(float t) {
//     if (t<0.)
//         return vec2(0.);
//     return vec2( tanh(sin(6.2831*30.0*t)*exp(-50.0*t)*10.0) );
// }



const float PI = 3.14159265358979323846;
const float TAU = 2*PI;

// Integrated sweep from f0 to 0 with decay rate kappa
// Also works for negative f0
 float intExpPhase(float t, float f0, float kappa){
    return (f0/kappa)*(1.0 - exp(-kappa*t));
}

// CREDIT: Hash functions from brainfiller/Ob5vr (https://github.com/0b5vr/brainfiller/blob/main/brainfiller.snd.glsl)
uvec3 hash3u(uvec3 v) {
  v = v * 1664525u + 1013904223u;

  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;

  v ^= v >> 16u;

  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;

  return v;
}

// float sinc(float x) { return abs(x) < 1e-7 ? 1.0 : sin(x) / x; }

// // For Blackman window, approx
// //    N = 17 -> -6 dB/oct
// //    N = 33 -> -12 dB/oct
// //    N = 65 -> -24 dB/oct
// float blackman(int n, int N){
//     float t = TAU*float(n)/(float(N)-1.0);
//     return 0.42 - 0.5*cos(t) + 0.08*cos(2.0*t);
// }

// float lp_tap(int k, int N, float fc) {
// 	float m = 0.5*(N-1), a = fc/SAMPLES_PER_SEC, x = float(k)-m;
// 	float hlp = 2.0*a*sinc(TAU*a*x);
// }

// float hp_tap(int k, int N, float fc) {
// 	// Spectral inversion
// 	return ((k==((N-1)/2)) ? 1.0 : 0.0) - lp_tap(k, N, fc);
// }

// float res_tap(int k, int N, float fc, float g) {
// 	float m = 0.5*(N-1), wc = TAU*fc/SAMPLES_PER_SEC, x = float(k)-m;
// 	return g * cos(wc*x);
// }

vec3 hash3f(vec3 v) {
  uvec3 u = hash3u(floatBitsToUint(v));
  return vec3(u) / float(-1u);
}

float env_curve(float edge0, float edge1, float x) {
	return smoothstep(edge0, edge1, x);
}

float env(float t, float attack, float decay) {
	if (t < 0. || t > (attack + decay))
		return 0.;
	if (t < attack) {
		return env_curve(0., attack, t);
	} else {
		return env_curve(attack + decay, attack, t);
	}
}

// More insp: https://www.youtube.com/watch?v=tPRBIBl5--w
vec2 kick(float t) {
    if (t<0.)
        return vec2(0.);
	float V = tanh(1.4*sin(TAU*(30.f*t + intExpPhase(t, 120.f, 10.f))) * exp(-t*10.));
	// float V = tanh(1.4*sin(TAU*(30.f*t + intExpPhase(t, 120.f, 10.f))) * env(t, 0.0, 0.35)); 
    return vec2(V);
}

// Inspiration: https://www.youtube.com/watch?v=tofBTvc3uT8
//              https://www.youtube.com/watch?v=Vr1gEf9tpLA
vec2 snare(float t) {
    if (t<0.)
        return vec2(0.);
    return vec2( sin(TAU*200.0*t)*env(t, 0.001, 0.15) );
}

vec2 hihat(float t, float step_) {
    if (t<0.)
        return vec2(0.);

	vec2 O = vec2(0.);

	repeat(i, 180) {
		vec3 r = hash3f(vec3(float(i + 11.9 + 1.*floor(step_))));
		// float freq = 8000.0 + 4000.0*(r.x*r.x*r.x);
		float freq = p2f(118. + 6.*r.x); // This is much nicer when distributed in pitch space
		float phase =  TAU*r.y;
		// float decay = (14.*(1.+.8*r.x)+5.*r.z);
		float decay = (14.*(1.+.8*r.x)+5.*r.z);
		float decay_s = 0.15*(1.+.3*r.x); // A bit more punchy
		// O += (1.0 - r.x*0.4)*vec2( sin(phase + TAU*freq*t) * exp(-decay*t) );
		O += (1.0 - r.x*0.4)*vec2( sin(phase + TAU*freq*t) * env(t, 5e-4, decay_s) );
	}

    // return vec2( 0.3*sin(TAU*1000.0*t)*exp(-3000.0*t) );
	return 0.02*O;
}

vec2 mainSound(int samp_in, float time_in) {
    vec4 time = vec4(samp_in % (SAMPLES_PER_BEAT * ivec4(1, 4, 64, 65536))) / SAMPLES_PER_SEC;
    vec4 beat = time*BPS;
  
    // A 440 Hz wave that attenuates  quickly overt time
    vec2 O = vec2(0.f);
    O += kick((beat.y-0.)*B2T);
    O += kick((beat.y-2.5)*B2T);
    O += snare((beat.y-1.)*B2T);
    O += snare((beat.y-3.)*B2T);
    O += hihat((beat.x-0.)*B2T, beat.x*2.);
    O += hihat((beat.x-0.5)*B2T, beat.x*2.);
	return O;
}

void main(){
	int offset = int(gl_GlobalInvocationID.x) + waveOutPosition;
	float sec = float(offset) / SAMPLES_PER_SEC;
	waveOutSamples[offset] = 1.*mainSound(offset, sec);

}

