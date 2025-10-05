#version 430
// CREDIT: Template from https://github.com/yosshin4004/minimal_gl/blob/master/examples/04_sound_output.snd.glsl, says "Copyright (C) 2020 Yosshin(@yosshin4004)"

// Arrangement
// https://www.youtube.com/watch?v=xY7vRHiSJSM <- Underdog on subtractive arrangement
// https://www.youtube.com/watch?v=oxJBrhyQ5jk <- Stranjah on the 16 bar block


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

// pub fn Pan(Pan_: f64, PanningLaw: f64) audio_types.stereo_sample {
//     // From http://music.columbia.edu/pipermail/music-dsp/2002-September/050872.html
//     const Scale = 2.0 - 4.0 * mathtiny.pow(10.0, PanningLaw / 20.0);
//     const PanR = Pan_;
//     const PanL = 1.0 - Pan_;
//     const GainL = Scale * PanL * PanL + (1.0 - Scale) * PanL;
//     const GainR = Scale * PanR * PanR + (1.0 - Scale) * PanR;
//     return audio_types.StereoSample(GainL, GainR);
// }

// CREDIT: http://music.columbia.edu/pipermail/music-dsp/2002-September/050872.html
vec2 pan(float pan, float law) {
	float Scale = 2.0-4.0*pow(10.0, law/20.0);
	vec2 Pan = vec2(pan, 1-pan);
	vec2 Gain = Scale*Pan*Pan + (1.0-Scale)*Pan;
	return Gain;
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

vec3 hash3f(vec3 v) {
  uvec3 u = hash3u(floatBitsToUint(v));
  return vec3(u) / float(-1u);
}

vec3 hash3f_normalized(vec3 v) {
	return hash3f(v) * 2.0 - 1.0;
}

float sinc(float x) { return abs(x) < 1e-7 ? 1.0 : sin(x) / x; }

// For Blackman window, approx  
//    N = 17 -> -6 dB/oct
//    N = 33 -> -12 dB/oct
//    N = 65 -> -24 dB/oct
float blackman(int k, int N){
    float t = TAU*float(k)/(float(N)-1.0);
    return 0.42 - 0.5*cos(t) + 0.08*cos(2.0*t);
}

float lp_tap(int k, int N, float fc) {
	float m = 0.5*(N-1), a = fc/SAMPLES_PER_SEC, x = float(k)-m;
	float hlp = 2.0*a*sinc(TAU*a*x);
	return hlp;
}

float hp_tap(int k, int N, float fc) {
	// Spectral inversion
	return ((k==((N-1)/2)) ? 1.0 : 0.0) - lp_tap(k, N, fc);
}

float res_tap(int k, int N, float fc, float g) {
	float m = 0.5*(N-1), wc = TAU*fc/SAMPLES_PER_SEC, x = float(k)-m;
	return g * cos(wc*x);
}

vec2 stereowidth(vec2 v, float w) {
	return mix(vec2((v.x+v.y)*0.5), v, vec2(w));
}

// More insp: https://www.youtube.com/watch?v=tPRBIBl5--w
// vec2 dirtykick(float t) {
//     if (t<0.)
//         return vec2(0.);
// 	vec2 V = vec2(tanh(1.4*sin(TAU*(hash3f(vec3(t)).x*0.155*exp(-t*8.0).x + 40.f*t + intExpPhase(t, 120.f, 10.f))) * exp(-t*6.)));
// 	V += hash3f(vec3(t)).xy*2.1*exp(-t*50.);
// 	float Drive = 2.0;
// 	V = tanh(V*Drive);
// 	V *= abs(sin(t*45.));
//     return V;
// }

vec2 dirtykick2(float t) {
    if (t<0.)
        return vec2(0.);
	vec2 V = vec2(tanh(1.4*sin(TAU*(hash3f_normalized(vec3(t)).x*0.035*exp(-t*10.0).x + 40.f*t + intExpPhase(t, 120.f, 10.f))) * exp(-t*6.)));
	V += hash3f_normalized(vec3(t)).xy*exp(-t*50.);
	float Drive = 2.0;
	V = tanh(V*Drive);
	V = stereowidth(V, 0.75);
	// V *= abs(sin(t*45.));
    return V;
}

vec2 kick(float t) {
    if (t<0.)
        return vec2(0.);
	vec2 V = vec2(tanh(1.4*sin(TAU*(40.f*t + intExpPhase(t, 120.f, 10.f))) * exp(-t*10.)));
	V += hash3f_normalized(vec3(t)).xy*exp(-t*50.);
	float Drive = 2.0;
	V = tanh(V*Drive);
	V *= abs(sin(t*15.)); // Ok, this is a little luck, gives a nice bounce
    return V;
}

// Inspiration: https://www.youtube.com/watch?v=tofBTvc3uT8
//              https://www.youtube.com/watch?v=Vr1gEf9tpLA
// vec2 snare(float t) {
//     if (t<0.)
//         return vec2(0.);
//     return vec2( sin(TAU*200.0*t)*env(t, 0.001, 0.15) );
// }

float linearenv_curve(float edge0, float edge1, float x) {
	return clamp((x - edge0) / (edge1 - edge0), 0., 1.);
}

float linearenv(float t, float attack, float decay) {
	if (t < 0. || t > (attack + decay))
		return 0.;
	if (t < attack) {
		return linearenv_curve(0., attack, t);
	} else {
		return 1.0 - linearenv_curve(attack, attack + decay, t);
	}
}

float linearenvwithhold(float t, float attack, float hold, float decay) {
	if (t < 0. || t > (attack + hold + decay)) {
		return 0.;
	} else if (t < attack) {
		return linearenv_curve(0., attack, t);
	} else if (t < attack + hold) {
		return 1.0;
	} else {
		return 1.0 - linearenv_curve(attack + hold, attack + hold + decay, t);
	}
}

float linearenvexp(float t, float attack, float kappa) {
	if (t < 0.) {
		return 0.;
	} else if (t < attack) {
		return linearenv_curve(0., attack, t);
	} else {
		return exp(-(t - attack)*kappa);
	}
}

vec2 snare2(float t, float time) {
	if (t<0.0)
		return vec2(0.0);
	// May not be necessary to filter this, it is very short	
	vec2 hit = vec2(0.0);
	const int N = 65;
	repeat(n, N) {
		float tap = hp_tap(n, N, 8000.0) * blackman(n, N);
		hit += hash3f_normalized(vec3(time + float(n)/SAMPLES_PER_SEC)).xy * vec2(tap);
	}
	// V *= vec2(exp(-t*15.));
	hit *= linearenvwithhold(t, 0.003, 0.005, 0.001); // 5ms   
	hit = stereowidth(hit, 0.5);

	float bodyphase = TAU*(160.f*t + 1.0*intExpPhase(t, 40.f, 15.0f));
	vec2 body = vec2(0.4 * sin(bodyphase));
	body *= linearenvwithhold(t, 0.010, 0.020, 0.100);
	body = tanh(hit + body*3.0);
	// vec2 body = vec2(0.4 * sin(TAU*(440.f*t + 44.0*(1.0 - exp(-10.0*t)))));


	vec2 noise = vec2(0.0);
	const int N2 = 21;
	repeat(n, N2) {
		float tap = res_tap(n, N, 5000.0, 0.9) * blackman(n, N);
		noise += hash3f_normalized(vec3(time + float(n)/SAMPLES_PER_SEC)).xy * vec2(tap);
	}
	// V *= vec2(exp(-t*15.));
	// noise *= linearenv(t, 0.040, 0.280);
	noise = stereowidth(noise, 0.3);
	noise *= linearenvexp(t, 0.035, 13.0);

	vec2 bell = vec2(0.);

	repeat(i, 12) {
		vec3 r = hash3f(vec3(float(i + 16.7)));
		// float freq = 8000.0 + 4000.0*(r.x*r.x*r.x);
		// float freq = p2f(60. + 30.*r.x); // This is much nicer when distributed in pitch space
		float freq = 160. + 40.*r.x;
		float ampl = exp(-3.0*r.z);
		float phase = TAU*r.y;
		// float decay = (14.*(1.+.8*r.x)+5.*r.z);
		// O += (1.0 - r.x*0.4)*vec2( sin(phase + TAU*freq*t) * exp(-decay*t) );
		bell += tanh(vec2( ampl*vec2(cos(phase + TAU*freq*t), sin(phase + TAU*freq*t)) )*2.0);
	}
	bell *= linearenvexp(t-0.015, 0.010, 10.0);
	bell = stereowidth(bell, 0.3);


	vec2 V = tanh((body*0.3 + noise*0.2 + bell*0.07)*5.0);
	// vec2 V = noise;

	return V;
}

// vec2 hihat(float t, float step_) {
//     if (t<0.)
//         return vec2(0.);

// 	vec2 O = vec2(0.);

// 	repeat(i, 180) {
// 		vec3 r = hash3f(vec3(float(i + 11.9 + 1.*floor(step_))));
// 		// float freq = 8000.0 + 4000.0*(r.x*r.x*r.x);
// 		float freq = p2f(118. + 6.*r.x); // This is much nicer when distributed in pitch space
// 		float phase =  TAU*r.y;
// 		// float decay = (14.*(1.+.8*r.x)+5.*r.z);
// 		float decay = (14.*(1.+.8*r.x)+5.*r.z);
// 		float decay_s = 0.15*(1.+.3*r.x); // A bit more punchy
// 		// O += (1.0 - r.x*0.4)*vec2( sin(phase + TAU*freq*t) * exp(-decay*t) );
// 		O += (1.0 - r.x*0.4)*vec2( sin(phase + TAU*freq*t) * env(t, 5e-4, decay_s) );
// 	}

//     // return vec2( 0.3*sin(TAU*1000.0*t)*exp(-3000.0*t) );
// 	return 0.02*O;
// }

// Needs freerunning time to avoid repetition, which is audible
vec2 hihat2(float t, float step_, float time) {
    if (t<0.)
        return vec2(0.);
	
	vec2 V = vec2(0.0);
	const int N = 65;
	repeat(n, N) {
		float tap = hp_tap(n, N, 8000.0) * blackman(n, N);
		V += hash3f_normalized(vec3(time + float(n)/SAMPLES_PER_SEC)).xy * vec2(tap);
	}

	V *= vec2(exp(-t*15.));
	// V *= env(t, 5e-2, 0.4);
	V = stereowidth(V, 0.75);
	return V*0.3;
}

vec2 bassfm(float t, float f0) {
	if (t<0.)
		return vec2(0.);
	
	return vec2(sin(TAU*f0*t + 2.1*sin(TAU*f0*2.01*t +  4.0*sin(TAU*f0*1.51*t + 2.5*sin(TAU*f0*2.0*t)))))*exp(-t*1.5);
}

vec2 bassfm2(float t, float f0) {
	vec2 O = vec2(0.0);
	const int N = 15;
	repeat(i, N) {
		vec2 V = bassfm(t+float(i)*0.001, f0 * exp(0.0005*(-(float(N-1)/2)+float(i))));
		V = V*pan(0.0 + (1.0/float(N-1))*float(i), -4.5);
		O += V;
	}
	return tanh(O);
}

vec2 bassfmqq(float t, float f0) {
	if (t<0.)
		return vec2(0.);
	
	return vec2(sin(TAU*f0*t + 2.1*sin(TAU*f0*4.01*t + 1.0*sin(TAU*f0*1.51*t + 3.5*sin(TAU*f0*2.0*t)))));
}

vec2 bassfm2qq(float t, float f0) {
	vec2 O = vec2(0.0);
	const int N = 15;
	repeat(i, N) {
		vec2 V = bassfmqq(t+float(i)*0.001, f0 * exp(0.0005*(-(float(N-1)/2)+float(i))));
		V = V*pan(0.0 + (1.0/float(N-1))*float(i), -4.5);
		O += V;
	}
	return tanh(O);
}

vec2 bassfm3(float t, float f0) {
	vec2 O = vec2(0.0);
	O += sin(TAU*f0*t) * linearenvwithhold(t, 0.005, 0.020, 0.100) ;
	return O;
}

vec2 bassline(vec4 time, vec4 beat) {
	#define BASS(offset, note) if (beat.z >= float(offset)) { noteFreq = p2f(float(note)); noteBeat = beat.z - float(offset); }
    float noteFreq 		= 0.0;
    float noteBeat		= 0.0;

	BASS(0, 41)
	BASS(1, 37)
	BASS(2, 43)
	BASS(3, 31)

	return 0.7* bassfm3(noteBeat*B2T, noteFreq);
	#undef BASS
}

vec2 saw(float t, float f0, float fc) {
	float V = 0.0;
	repeat(i, 127) {
		float f = f0 * float(i+1);
		if (f < 16000.0) {
			float a = 1.0 / float(i+1);
			if (f > fc) {
				float reloct = f / fc;
				a *= 1.0 / (1.0 + pow(reloct, 2.0));
			}
			V += a * sin(TAU * f * t);
		}
	}
	return vec2(V);
}

vec2 reese(float t, float f0, float fc) {
	vec2 O = vec2(0.0);
	repeat(i, 9) {
		float j = i - 4.0;
		float f = f0 * exp(j*0.005);
		vec2 V = saw(t, f, fc) * (1.0 / (abs(j)*0.5 + 1.0));
		O += vec2(V) * pan(0.5 + float(j)*0.1, -4.5);
	}
	return O;
}

float tri(float x) {
	float x2 = mod(x, 1.0) * 2.0;
	if (x2 > 1.0)
		return 1.0 - x2;
	return x2 ;
}

// Inspiration: https://www.youtube.com/shorts/pYKKurirXV0
vec2 noisebass(float t, float f0) {
	float b = t * T2B;
	vec2 V = sin(t*TAU*f0) + 0.08*hash3f_normalized(vec3(t + 0.123)).xy;
	V *= abs(tri(b));
	V = stereowidth(V, 0.5);

	return vec2(tanh(V*2.0));
}

// Inspiration: https://www.youtube.com/watch?v=BxehYL9Abg4
vec2 shaker(float t) {
    if (t<0.)
        return vec2(0.);
	
	vec2 V = vec2(0.0);
	const int N = 129;
	repeat(n, N) {
		float tap = hp_tap(n, N, 5000.0) * blackman(n, N);
		V += hash3f_normalized(vec3(t + float(n)/SAMPLES_PER_SEC + 0.997)).xy * vec2(tap);
	}

	float env = 0.0
	+ 0.8*linearenv(t, 0.005, 0.105)
	+ 0.1*linearenv(t-0.110, 0.015, 0.110)
	+ 0.2*linearenv(t-0.235, 0.010, 0.140)
	;

	V *= env;
	// V *= env(t, 5e-2, 0.4);
	V = stereowidth(V, 0.60);
	return V*0.3;
}

vec2 riser(float t) {
    if (t<0.)
        return vec2(0.);
	
	float p0 = 25.0 + t*25.0;

	vec2 V = vec2(0.0);
	const int N = 221 ;
	repeat(n, N) {
		vec3 r = hash3f(vec3(n*1.1 + 0.9128783));
		float p = 20.0 + 80.0 * r.x;
		float a = exp(-0.2*pow(p0 - p, 2.0));
		float Q = a * sin(TAU * p2f(p) * t + r.z*TAU);
		V += Q * pan(r.y, -4.5);
	}

	V = stereowidth(V, 0.25);
	return tanh(V)*0.2;
}

vec2 mainbass_bar(float barpos) {
	float note = 40.0;

	float curnote, curpan, curvol;

	float pl=0.1,pr=0.9;

#define B(pos, note, pan, vol) if (barpos > pos) { curnote=note; curpan=pan; curvol=vol; }
	B(0.0, note, pl, 1.0)
	B(0.5, note, pr, 1.0)
	B(1.0, note, pl, 0.0)
	B(1.5, note, pr, 1.0)
	B(2.0, note, pl, 1.0)
	B(2.5, note, pr, 1.0)
	B(3.0, note, pl, 0.5)
	B(3.5, note, pr, 1.0)
#undef B

	return 2.0*(0.7*exp(-4.0*mod(barpos, 0.50))*bassfm2(barpos*B2T, p2f(curnote))*pan(curpan, -4.5))*curvol;
}

vec2 gnarly1bass_bar(float barpos) {
	float note = 36.0;

	float curnote, curpan, curvol;

	// TODO: Cleanup pan 
	float pl=0.5,pr=0.5;

#define B(pos, note, pan, vol) if (barpos > pos) { curnote=note; curpan=pan; curvol=vol; }
	B(0.0, note, pl, 1.0)
	B(1.0, note, pl, 1.0)
	B(2.0, note, pl, 1.0)
	B(3.0, note, pl, 1.0)
#undef B

	return exp(-5.0*mod(barpos, 1))*bassfm2qq(barpos*B2T, p2f(curnote))*curvol; // Go to time.x for energy

}

vec2 noisebass_bar(float barpos) {
	return noisebass(barpos*B2T, p2f(40.0));
}

vec2 gnarly2bass_bar(float barpos) {
	float beatpos = mod((barpos * 4.0), 1.0);
	vec2 O = vec2(0.);
	O += 1.0*exp(-4.0*beatpos*B2T)*bassfm2qq(barpos*B2T, p2f(36.0)); // Go to time.x for energy
	O += 1.0*exp(-4.0*(beatpos-0.5)*B2T)*bassfm2qq(mod(barpos-0.5, 4.0)*B2T, p2f(28.0)); // Go to time.x for energy
	return O;
}

vec2 mainSound(int samp_in, float time_in) {
    vec4 time = vec4(samp_in % (SAMPLES_PER_BEAT * ivec4(1, 4, 64, 65536))) / SAMPLES_PER_SEC;
    vec4 beat = time*BPS;
  
	float barpos = mod(beat.z, 4.0);

	float altbar = 0.0;
	if ((beat.z >= 24.0 && beat.z < 32.0) || (beat.z >= 56.0 && beat.z < 64.0)) {
		altbar = 1.0;
	}

    // A 440 Hz wave that attenuates quickly over time
    vec2 O = vec2(0.f);
	if (true) {
		O += kick((beat.y-0.)*B2T);
		O += dirtykick2((beat.y-2.5)*B2T);
		O += snare2((beat.y-1.)*B2T, time.w + 0.789);
		// O += snare2((beat.y-3.)*B2T, time.w + 0.451);
		// // O += hihat2((beat.x-0.0)*B2T, beat.x*2.);

		// O += hihat2((beat.x-0.25)*B2T, beat.x*2., time.y + 0.123);
		// O += hihat2((beat.x-0.0)*B2T, beat.x*2., time.w + 0.456); // Time wraps at the end...

		// Another possibility - part of this is cutoff because it doesn't wrap time, so just cuts off
		if (altbar == 0.f) {
			O += 1.5*hihat2((beat.x-0.75+1.0)*B2T, beat.x*2., time.y + 0.123);
			O += 1.5*hihat2((beat.x-0.50)*B2T, beat.x*2., time.w + 0.456); // Time wraps at the end...
		} else {
			if (beat.z < 32.0) {
				O += shaker(time.x)*2.0;
			} else {
				O += riser(mod(beat.z, 8.0)*B2T) * 1.4;
			}
		}

		// O += hihat((beat.x-0.50)*B2T, beat.x*2.);
		// O += hihat((beat.x-0.75)*B2T, beat.x*2.);
	}

	float percsidechain =
		  1.0*linearenvwithhold((beat.y-2.5)*B2T - 0.050, 0.050, 0.250, 0.100)
		+ 0.8*linearenvwithhold((beat.y-0.0)*B2T - 0.000, 0.000, 0.150, 0.100)
		+ 0.7*linearenvwithhold((beat.y-1.0)*B2T - 0.000, 0.010, 0.100, 0.100)
		;
	percsidechain = 1.0 - tanh(percsidechain*1.5 );

	if (true) {
		if (altbar == 0.f) {
			if (mod(beat.z, 8.0) < 4.0) {
				// main bass
				// if (beat.y < 0.50)
				// 	note = 1.0;
				O += mainbass_bar(barpos) * percsidechain; // Go to time.x for energy
			} else {
				// alt bass
				float bassindex = floor(mod(beat.z, 32.0) / 8.0);

				if (bassindex == 0.0) {
					// gnarly 1
					O += 1.7*gnarly1bass_bar(barpos) * percsidechain;
					// O += 1.7*exp(-5.0*mod(beat.x, 1))*bassfm2qq(mod(beat.z, 4.0)*B2T, p2f(36.0)) * percsidechain; // Go to time.x for energy
				} else if (bassindex == 1.0) {
					// noise bass
					// O += noisebass(time.y, p2f(40.0)) * percsidechain;
					O += noisebass_bar(barpos) * percsidechain;
				} else {
					// gnarly 2
					O += 1.0*exp(-4.0*beat.y*B2T)*bassfm2qq(barpos*B2T, p2f(36.0)) * percsidechain; // Go to time.x for energy
					O += 1.0*exp(-4.0*(beat.y-0.5)*B2T)*bassfm2qq(mod(barpos-0.5, 4.0)*B2T, p2f(28.0)) * percsidechain; // Go to time.x for energy
				} 
					

			}
			// gnarly 1
			// if (mod(beat.z, 8.0) >= 4.0) {
			// }
			// // gnarly 2
			// if (mod(beat.z, 8.0) >= 4.0) {
			// 	O += 1.7*exp(-5.0*mod(beat.x, 1.50))*bassfm2qq(mod(beat.z, 4.0)*B2T, p2f(36.0)) * percsidechain; // Go to time.x for energy
			// }
		} else {
			// reese bass
			float time = mod(beat.z, 8.0)*B2T;
			float length = SAMPLES_PER_BEAT*8.0 / SAMPLES_PER_SEC;
			float offset = 0.2;
			O += 0.6 * reese(time, p2f(37.0), p2f(50.0 + time * 13.0)) * linearenvwithhold(time - offset, 0.300, length - 0.500 - offset, 0.200) * percsidechain;
		}
	}

	// O = vec2(0.0);

	// O += mainbass_bar(barpos);
	// O += gnarly1bass_bar(barpos);
	// O += gnarly2bass_bar(barpos);
	// O += noisebass_bar(barpos);

	// O += riser(mod(beat.z, 8.0)*B2T);
	// O += riser(mod(beat.z, 8.0)*B2T);

	// O += noisebass(time.y, p2f(40.0)) * 0.3;

	// O += shaker((beat.y-1.0)*B2T);
	// O += shaker((beat.y-2.0)*B2T);
	// O += shaker((beat.y-3.0)*B2T);

	// O += bassline(time, beat);

	return 1.0*clamp(O, -1.0, 1.0);
}

void main(){
	int offset = int(gl_GlobalInvocationID.x) + waveOutPosition;
	float sec = float(offset) / SAMPLES_PER_SEC;
	waveOutSamples[offset] = 1.*mainSound(offset, sec);

}

