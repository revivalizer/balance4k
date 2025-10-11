 #version 430
// CREDIT: Template from https://github.com/yosshin4004/minimal_gl/blob/master/examples/04_sound_output.snd.glsl, says "Copyright (C) 2020 Yosshin(@yosshin4004)"

// Arrangement
// https://www.youtube.com/watch?v=xY7vRHiSJSM <- Underdog on subtractive arrangement
// https://www.youtube.com/watch?v=oxJBrhyQ5jk <- Stranjah on the 16 bar block

// CREDIT: FROM brainfiller/Ob5vr (https://github.com/0b5vr/brainfiller/blob/main/brainfiller.snd.glsl)
#define repeat(i, n) for(int i=0; i<(n); i++)
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

float log10(float v) {
    return log(v) / log(10.0);
}

float to_db(float gain) {
    return 20.0 * log10(gain);
}

float from_db(float db) {
    return pow(10.0, db / 20.0);
}

layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
#pragma work_around_begin:layout(std430,binding=0)buffer ssbo{vec2 %s[];};layout(local_size_x=1)in;
vec2 waveOutSamples[];
#pragma work_around_end
#else
layout(std430, binding = 0) buffer SoundOutput {
    vec2 waveOutSamples[];
};
layout(local_size_x = 1) in;
#endif

const float BPM = 170;
const float BPS = BPM / 60.f;

const float SAMPLES_PER_SEC = 48000.; // 48000 in minimal_gl

const int SAMPLES_PER_STEP = int(SAMPLES_PER_SEC / BPS / 4.0);
const int SAMPLES_PER_BEAT = 4 * SAMPLES_PER_STEP;

const float T2B = BPS;
const float B2T = 1.0 / BPS;
//const float S2T = 0.25 * B2T;

// vec2 kick(float t) {
//     if (t < 0.)
//         return vec2(0.);
//     return vec2(tanh(sin(6.2831 * 30.0 * t) * exp(-50.0 * t) * 10.0));
// }

const float PI = 3.14159265358979323846;
const float TAU = 2 * PI;

// Integrated sweep from f0 to 0 with decay rate kappa
// Also works for negative f0
float intExpPhase(float t, float f0, float kappa) {
    return (f0 / kappa) * (1.0 - exp(-kappa * t));
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
    float Scale = 2.0 - 4.0 * pow(10.0, law / 20.0);
    vec2 Pan = vec2(pan, 1 - pan);
    vec2 Gain = Scale * Pan * Pan + (1.0 - Scale) * Pan;
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

float sinc(float x) {
    return abs(x) < 1e-7 ? 1.0 : sin(x) / x;
}

// For Blackman window, approx
//    N = 17 -> -6 dB/oct
//    N = 33 -> -12 dB/oct
//    N = 65 -> -24 dB/oct
float blackman(int k, int N) {
    float t = TAU * float(k) / (float(N) - 1.0);
    return 0.42 - 0.5 * cos(t) + 0.08 * cos(2.0 * t);
}

float lp_tap(int k, int N, float fc) {
    float m = 0.5 * (N - 1), a = fc / SAMPLES_PER_SEC, x = float(k) - m;
    float hlp = 2.0 * a * sinc(TAU * a * x);
    return hlp;
}

float hp_tap(int k, int N, float fc) {
    // Spectral inversion
    return ((k == ((N - 1) / 2)) ? 1.0 : 0.0) - lp_tap(k, N, fc);
}

float res_tap(int k, int N, float fc, float g) {
    float m = 0.5 * (N - 1), wc = TAU * fc / SAMPLES_PER_SEC, x = float(k) - m;
    return g * cos(wc * x);
}

vec2 stereowidth(vec2 v, float w) {
    return mix(vec2((v.x + v.y) * 0.5), v, vec2(w));
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
    if (t < 0.)
        return vec2(0.);
    vec2 V = vec2(tanh(1.4 * sin(TAU * (hash3f_normalized(vec3(t)).x * 0.035 * exp(-t * 10.0).x + 40.f * t + intExpPhase(t, 120.f, 10.f))) * exp(-t * 6.)));
    V += hash3f_normalized(vec3(t)).xy * exp(-t * 50.);
    float Drive = 2.0;
    V = tanh(V * Drive);
    V = stereowidth(V, 0.75);
    // V *= abs(sin(t*45.));
    return V;
}

vec2 kick(float t) {
    if (t < 0.)
        return vec2(0.);
    vec2 V = vec2(tanh(1.4 * sin(TAU * (40.f * t + intExpPhase(t, 120.f, 10.f))) * exp(-t * 10.)));
    V += hash3f_normalized(vec3(t)).xy * exp(-t * 50.);
    float Drive = 2.0;
    V = tanh(V * Drive);
    V *= abs(sin(t * 15.)); // Ok, this is a little luck, gives a nice bounce
    return V;
}

float linearenv_curve(float edge0, float edge1, float x) {
    return clamp((x - edge0) / (edge1 - edge0), 0., 1.);
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

float linearenv(float t, float attack, float decay) {
    return linearenvwithhold(t, attack, 0.0, decay);
}

float linearenvexp(float t, float attack, float kappa) {
    if (t < 0.) {
        return 0.;
    } else if (t < attack) {
        return linearenv_curve(0., attack, t);
    } else {
        return exp(-(t - attack) * kappa);
    }
}

// Inspiration: https://www.youtube.com/watch?v=tofBTvc3uT8
//              https://www.youtube.com/watch?v=Vr1gEf9tpLA
vec2 snare2(float t, float time) {
    if (t < 0.0)
        return vec2(0.0);
    if (t > 0.5)
        return vec2(0.0);

    // May not be necessary to filter this, it is very short
    vec2 hit = vec2(0.0);
    const int N = 65;
    repeat(n, N)
    {
        float tap = hp_tap(n, N, 8000.0) * blackman(n, N);
        hit += hash3f_normalized(vec3(time + float(n) / SAMPLES_PER_SEC)).xy * vec2(tap);
    }
    // V *= vec2(exp(-t*15.));
    hit *= linearenvwithhold(t, 0.003, 0.005, 0.001); // 5ms
    hit = stereowidth(hit, 0.5);

    float bodyphase = TAU * (160.f * t + 1.0 * intExpPhase(t, 40.f, 15.0f));
    vec2 body = vec2(0.4 * sin(bodyphase));
    body *= linearenvwithhold(t, 0.010, 0.020, 0.100);
    body = tanh(hit + body * 3.0);
    // vec2 body = vec2(0.4 * sin(TAU*(440.f*t + 44.0*(1.0 - exp(-10.0*t)))));

    vec2 noise = vec2(0.0);
    const int N2 = 21;
    repeat(n, N2)
    {
        float tap = res_tap(n, N, 5000.0, 0.9) * blackman(n, N);
        noise += hash3f_normalized(vec3(time + float(n) / SAMPLES_PER_SEC)).xy * vec2(tap);
    }
    // V *= vec2(exp(-t*15.));
    // noise *= linearenv(t, 0.040, 0.280);
    noise = stereowidth(noise, 0.3);
    noise *= linearenvexp(t, 0.035, 13.0);

    vec2 bell = vec2(0.);

    repeat(i, 12)
    {
        vec3 r = hash3f(vec3(float(i + 16.7)));
        // float freq = 8000.0 + 4000.0*(r.x*r.x*r.x);
        // float freq = p2f(60. + 30.*r.x); // This is much nicer when distributed in pitch space
        float freq = 160. + 40. * r.x;
        float ampl = exp(-3.0 * r.z);
        float phase = TAU * r.y;
        // float decay = (14.*(1.+.8*r.x)+5.*r.z);
        // O += (1.0 - r.x*0.4)*vec2( sin(phase + TAU*freq*t) * exp(-decay*t) );
        bell += tanh(vec2(ampl * vec2(cos(phase + TAU * freq * t), sin(phase + TAU * freq * t))) * 2.0);
    }
    bell *= linearenvexp(t - 0.015, 0.010, 10.0);
    bell = stereowidth(bell, 0.3);

    vec2 V = tanh((body * 0.3 + noise * 0.2 + bell * 0.07) * 5.0);
    // vec2 V = noise;

    return V;
}

// Needs freerunning time to avoid repetition, which is audible
vec2 hihat2(float t, float step_, float time) {
    if (t < 0.)
        return vec2(0.);

    vec2 V = vec2(0.0);
    const int N = 65;
    repeat(n, N)
    {
        float tap = hp_tap(n, N, 8000.0) * blackman(n, N);
        V += hash3f_normalized(vec3(time + float(n) / SAMPLES_PER_SEC)).xy * vec2(tap);
    }

    V *= vec2(exp(-t * 15.));
    // V *= env(t, 5e-2, 0.4);
    V = stereowidth(V, 0.75);
    return V * 0.3;
}

// vec2 saw(float t, float f0, float fc) {
//     float V = 0.0;
//     repeat(i, 127)
//     {
//         float f = f0 * float(i + 1);
//         if (f < 16000.0) {
//             float a = 1.0 / float(i + 1);
//             if (f > fc) {
//                 float reloct = f / fc;
//                 a *= 1.0 / (1.0 + pow(reloct, 2.0));
//             }
//             V += a * sin(TAU * f * t);
//         }
//     }
//     return vec2(V);
// }

// vec2 reese(float t, float f0, float fc) {
//     vec2 O = vec2(0.0);
//     repeat(i, 9)
//     {
//         float j = i - 4.0;
//         float f = f0 * exp(j * 0.005);
//         vec2 V = saw(t, f, fc) * (1.0 / (abs(j) * 0.5 + 1.0));
//         O += vec2(V) * pan(0.5 + float(j) * 0.1, -4.5);
//     }
//     return O;
// }

// float tri(float x) {
//     float x2 = mod(x, 1.0) * 2.0;
//     if (x2 > 1.0)
//         return 1.0 - x2;
//     return x2;
// }

// // Inspiration: https://www.youtube.com/shorts/pYKKurirXV0
// vec2 noisebass(float t, float f0) {
//     float b = t * T2B;
//     vec2 V = sin(t * TAU * f0) + 0.08 * hash3f_normalized(vec3(t + 0.123)).xy;
//     V *= abs(tri(b));
//     V = stereowidth(V, 0.5);

//     return vec2(tanh(V * 2.0));
// }

// Inspiration: https://www.youtube.com/watch?v=BxehYL9Abg4
vec2 shaker(float t) {
    if (t < 0.)
        return vec2(0.);

    vec2 V = vec2(0.0);
    const int N = 129;
    repeat(n, N)
    {
        float tap = hp_tap(n, N, 5000.0) * blackman(n, N);
        V += hash3f_normalized(vec3(t + float(n) / SAMPLES_PER_SEC + 0.997)).xy * vec2(tap);
    }

    float env = 0.0
            + 0.8 * linearenv(t, 0.005, 0.105)
            + 0.1 * linearenv(t - 0.110, 0.015, 0.110)
            + 0.2 * linearenv(t - 0.235, 0.010, 0.140);

    V *= env;
    // V *= env(t, 5e-2, 0.4);
    V = stereowidth(V, 0.60);
    return V * 0.3;
}

vec2 riser2(float t) {
    if (t < 0.)
        return vec2(0.);

    float minp = 85.0;
    float maxp = 130.0;
    // float length = t * T2B / 8.0;
    float pfilter = minp + t / 2.7 * (maxp - minp);

    float env = t / 2.7;
    env = env * env * env;

    vec2 V = vec2(0.0);
    const int N = 221;
    repeat(n, N)
    {
        vec3 r = hash3f(vec3(n * 1.1 + 11.9128783));
        float p = minp + (maxp - minp) * r.x;
        float a = exp(-0.005 * pow(pfilter - p, 2.0));
        float magnitude = 1.0 / (2.0 * p2f(p) / p2f(minp));
        a *= magnitude;
        // if (p < pfilter)
        //     a = 1;

        float Q = a * sin(TAU * p2f(p) * t + r.z * TAU);
        V += Q * pan(r.y, -4.5);
    }

    V = env * stereowidth(V, 0.25);
    return V * 0.4;
}

// TODO: Merge with riser2
vec2 sweep(float t) {
    if (t < 0. || t > 8.0)
        return vec2(0.);

    float startp = 130.0;
    float stopp = 85.0;

    float minp = min(startp, stopp);
    float maxp = max(startp, stopp);
    // float length = t * T2B / 8.0;

    float time_factor = 2.7 * 1.0;

    float pfilter = startp + t / time_factor * (stopp - startp);

    float env = t / time_factor;
    env = exp(-t * time_factor * 0.5);

    vec2 V = vec2(0.0);
    const int N = 221;
    repeat(n, N)
    {
        vec3 r = hash3f(vec3(n * 1.1 + 11.9128783));
        float p = minp + (maxp - minp) * r.x;
        float a = exp(-0.005 * pow(pfilter - p, 2.0));
        float magnitude = 1.0 / (2.0 * p2f(p) / p2f(minp));
        a *= magnitude;
        // if (p < pfilter)
        //     a = 1;

        float Q = a * sin(TAU * p2f(p) * t + r.z * TAU);
        V += Q * pan(r.y, -4.5);
    }

    V = env * stereowidth(V, 0.25);
    return V * 0.4;
}

vec2 pad(float barpos, float note) {
    float t = barpos * B2T;
    // barpos goes from 0-64
    if (t < 0.)
        return vec2(0.);

    float note_freq = p2f(note);

    vec2 V = vec2(0.0);

    const int NUM_MODES = 30;
    repeat(mode, NUM_MODES)
    {
        vec3 mode_r = hash3f_normalized(vec3(mode * 13.1 + 9.9128783));
        float mode_magnitude = pow(2.0, (1.5 + 3.0 * mode / float(NUM_MODES)) * mode_r.x * sin(TAU * (0.02 + 0.13 * mode_r.y) * t + mode_r.z));

        const int NUM_HARMONICS_PER_MODE = 19;
        repeat(harmonic, NUM_HARMONICS_PER_MODE)
        {
            vec3 r = hash3f_normalized(vec3(mode * 1.1 + 11.9128783 + harmonic * 1.9));

            float mode_freq = note_freq * (mode + 1.0);
            float rel_oct = log(mode_freq / note_freq);

            float freq = mode_freq * pow(2.0, sqrt(sqrt(rel_oct + 2.0)) * 0.02 * r.x); // scale up with freq

            float magnitude = 1.0 / pow((mode_freq / note_freq), 1.5);

            // if (p < pfilter)
            //     a = 1;

            float Q = mode_magnitude * magnitude * sin(TAU * freq * t + r.z * TAU);
            V += Q * pan(r.y * 0.5 + 0.5, -4.5);
        }
    }

    float env = linearenvwithhold(barpos, 4.0, 52.0, 8.0);

    V = env * stereowidth(V, 1.0);
    return V * 0.04;
}

float filterformant(float freq, vec3 formant) {
    float f = formant.x;
    float gain = formant.y;
    float bw = formant.z * 2.0;
    float bw2 = bw * bw;

    // float V = linearenv_curve(f - bw, f, freq) * linearenv_curve(f, f + bw, freq); // Tri shape at f, +/- bw
    float V = exp(-pow(freq - f, 2.0) / (2.0 * bw2));
    V *= gain;
    return V;
}

// // CREDIT: Formant values: https://www.classes.cs.uchicago.edu/archive/1999/spring/CS295/Computing_Resources/Csound/CsManual3.48b1.HTML/Appendices/table3.html
// const vec3 formant_tenor_a[5] = vec3[](
//         // vec3(650, from_db(0), 80),
//         // vec3(1080, from_db(-6), 90),
//         // vec3(2650, from_db(-7), 120),
//         // vec3(2900, from_db(-8), 130),
//         // vec3(3250, from_db(-22), 140)
//         // CREDIT: ChatGPT did these for me
//         vec3(650, 1.00, 80),
//         vec3(1080, 0.50, 90),
//         vec3(2650, 0.45, 120),
//         vec3(2900, 0.40, 130),
//         vec3(3250, 0.08, 140)
//     );

// // // const vec3 formant_bass_a[5] = vec3[](
// // //         vec3(600, from_db(0), 80),
// // //         vec3(1040, from_db(-7), 90),
// // //         vec3(2250, from_db(-9), 120),
// // //         vec3(2450, from_db(-9), 130),
// // //         vec3(2750, from_db(-20), 140)
// // //     );

// // // const vec3 formant_tenor_u[5] = vec3[](
// // //         vec3(350, from_db(0), 40),
// // //         vec3(600, from_db(-20), 60),
// // //         vec3(2700, from_db(-17), 100),
// // //         vec3(2900, from_db(-14), 120),
// // //         vec3(3300, from_db(-26), 120)
// // //     );

// const vec3 formant_tenor_o[5] = vec3[](
//         // vec3(400, from_db(0), 40),
//         // vec3(800, from_db(-10), 80),
//         // vec3(2600, from_db(-12), 100),
//         // vec3(2800, from_db(-12), 120),
//         // vec3(3000, from_db(-26), 120)
//         // CREDIT: ChatGPT also did these for me. Blind trust.
//         vec3(400, 1.00, 40),
//         vec3(800, 0.32, 80),
//         vec3(2600, 0.25, 100),
//         vec3(2800, 0.25, 120),
//         vec3(3000, 0.05, 120)
//     );

// NOTE: Shader minifier has problems with the above construction
//  This is formant_tenor_a and formant_tenor_o
const vec3 formant_table[10] = vec3[](
        vec3(650, 1.00, 80),
        vec3(1080, 0.50, 90),
        vec3(2650, 0.45, 120),
        vec3(2900, 0.40, 130),
        vec3(3250, 0.08, 140),
        vec3(400, 1.00, 40),
        vec3(800, 0.32, 80),
        vec3(2600, 0.25, 100),
        vec3(2800, 0.25, 120),
        vec3(3000, 0.05, 120)
    );

vec2 pad3voice(float barpos, float barnum, float note) {
    float t = barpos * B2T;
    // barpos goes from 0-64
    if (t < 0.)
        return vec2(0.);
    if (barpos > 1.0) {
        return vec2(0.); // Optimization
    }

    float note_freq = p2f(note);

    vec2 V = vec2(0.0);

    const int NUM_MODES = 30;
    repeat(mode, NUM_MODES)
    {
        vec3 mode_r = hash3f_normalized(vec3(mode * 13.1 + 9.9128783 + barnum * 0.1432));
        float mode_magnitude = pow(2.0, 1.07 * mode_r.x * sin(TAU * (0.02 + 0.13 * mode_r.y) * t + mode_r.z));

        const int NUM_HARMONICS_PER_MODE = 19 * 2;
        repeat(harmonic, NUM_HARMONICS_PER_MODE + int(mode))
        {
            vec3 r = hash3f_normalized(vec3(mode * 1.1 + 11.9128783 + harmonic * 1.9 + barnum * 0.1432));

            float mode_freq = note_freq * (mode + 1.0);
            float rel_oct = log(mode_freq / note_freq);

            float freq = mode_freq * pow(2.0, sqrt(sqrt(rel_oct + 2.0)) * 0.027 * r.x); // scale up with freq

            float magnitude = 1.0 / pow((mode_freq / note_freq), 1.5);

            float formant_glide = smoothstep(0.0 - abs(r.x) * 0.05, 0.5 + r.y * 0.03, barpos);

            float formant = 0.0;
            repeat(n, 5)
            {
                formant = max(formant, filterformant(freq, mix(formant_table[n + 5], formant_table[n], formant_glide)));
            }

            magnitude *= 0.0 + 1.0 * formant;

            float env = linearenvwithhold(barpos, 0.15 + abs(r.x) * 0.01, 0.3 + abs(r.y) * 0.003, 0.5 + abs(r.z) * 0.05);
            magnitude *= env;

            float Q = mode_magnitude * magnitude * sin(TAU * freq * t + r.z * TAU);
            V += Q * pan(r.y * 0.5 + 0.5, -4.5);
        }
    }

    V = stereowidth(V, 0.6);
    return V * 0.4;
}

// IDEAS: Gatet effekt paa wah
//        Crowds ?
//        Ekstra snare paa hver 4 beat (sidste)

float beat_comp(float block, float bar, float beat) {
    return (block * 16.0 + bar) * 4.0 + beat;
}

vec2 mainSound(int samp_in, float time_in) {
    const int InitialQuietSamples = int(SAMPLES_PER_SEC * 0.5);
    if (samp_in < InitialQuietSamples)
        return vec2(0.0);

    // int samp_offset = 0;
    int samp_offset = (SAMPLES_PER_BEAT * 4) * 0;

    vec4 time = vec4((samp_in + samp_offset - InitialQuietSamples) % (SAMPLES_PER_BEAT * ivec4(1, 4, 64, 65536))) / SAMPLES_PER_SEC;
    vec4 beat = time * BPS;

    float block = floor(beat.w / 64.0);
    float half_block = floor(beat.w / 32.0);
    float bar_in_block_unfloor = floor(beat.z / 4.0);
    float bar_in_block = floor(bar_in_block_unfloor);

    float is_intro_1 = block == 0 ? 1.0 : 0.0;
    float is_intro_2 = block == 1 ? 1.0 : 0.0;
    // float is_intro = is_intro_1 + is_intro_2;

    // float enable_bass = block >= 2.0 ? 1.0 : 0.0;
    float enable_bass = 0.0;
    float enable_pad = block < 5.0 ? 1.0 : 0.0;

    // float enable_wah = ((block == 1.0 && bar_in_block_unfloor >= 14.5) || (block >= 2.0 && (block <= 5.0 || (block == 5.0 && bar_in_block_unfloor < 2.25)))) ? 1.0 : 0.0;
    float enable_wah = (beat.w >= beat_comp(1.0, 14.5, 0.0) && beat.w < beat_comp(5.0, 2.25, 0.0)) ? 1.0 : 0.0;
    float enable_kick = (block >= 1.0 && block <= 4.0) ? 1.0 : 0.0;
    float enable_snare = ((block >= 1.0 && block != 3.0 && block <= 4.0) ? 1.0 : 0.0);
    float enable_snare2 = enable_snare * ((block >= 4.0 && block <= 6.0) ? 1.0 : 0.0);
    float enable_hihat = (block < 5.0) ? 1.0 : 0.0;
    float enable_sweep = block >= 1.0 && block < 5.0 ? 1.0 : 0.0;
    // float enable_kick = block > 0.

    float enable_block_last_bar = (block == 1.0 && bar_in_block_unfloor >= 14.0) ? 0.0 : 1.0;
    enable_kick *= enable_block_last_bar;
    enable_snare *= enable_block_last_bar;
    enable_hihat *= enable_block_last_bar;

    float barpos = mod(beat.z, 4.0);

    bool altbar = false;
    if ((beat.z >= 24.0 && beat.z < 32.0) || (beat.z >= 56.0 && beat.z < 64.0)) {
        altbar = true;
    }

    // A 440 Hz wave that attenuates quickly over time
    // vec2 O = vec2(sin(time.z * TAU * 440.0) * 0.1);
    vec2 O = vec2(0.);

    if (true) {
        float drum_gain = 0.85;
        O += enable_kick * kick((beat.y - 0.) * B2T) * drum_gain;
        O += enable_kick * dirtykick2((beat.y - 2.5) * B2T) * drum_gain;
        O += enable_snare * snare2((beat.y - 1.) * B2T, time.w + 0.789) * drum_gain;
        // O += enable_snare2 * snare2((beat.y - 3.) * B2T, time.w + 0.451);

        // Another possibility - part of this is cutoff because it doesn't wrap time, so just cuts off
        if ((altbar == false || beat.w < 32.0) && beat.w > 32.0) {
            O += enable_hihat * 1.0 * hihat2((beat.x - 0.75 + 1.0) * B2T, beat.x * 2., time.y + 0.123);
            O += enable_hihat * 1.0 * hihat2((beat.x - 0.50) * B2T, beat.x * 2., time.w + 0.456); // Time wraps at the end...
        } else {
            if (block < 5.0) {
                if (beat.z < 32.0) {
                    if (block >= 1.0) { // We don't want riser in intro
                        O += shaker(time.x) * 1.2;
                    }
                } else {
                    O += riser2(mod(beat.z, 8.0) * B2T) * 0.7;
                }
            }
        }
    }

    float percsidechain =
        1.0 * linearenvwithhold((beat.y - 2.5) * B2T - 0.050, 0.050, 0.250, 0.100)
            + 0.8 * linearenvwithhold((beat.y - 0.0) * B2T - 0.000, 0.010, 0.140, 0.100)
            + 0.7 * linearenvwithhold((beat.y - 1.0) * B2T - 0.000, 0.010, 0.100, 0.100);
    percsidechain = 1.0 - tanh(percsidechain * 1.5);

    // O = vec2(0.0);
    O += snare2((beat.w - 63.5) * B2T, time.w + 0.789);

    float pad_switch_beat = mod(beat.z - 24.0, 32.0); // Last 2 bar in half block
    float pad_switch_env =
        (half_block >= 5.0 && half_block <= 7.0) ? linearenvwithhold(pad_switch_beat + 0.6, 0.6, 8.0, 0.6) : 0.0;

    if ((1.0 - pad_switch_env) > 0.0) {
        O += (1.0 - pad_switch_env) * enable_pad * 1.5 * pad(mod(beat.z, 64.0), 40.0) * sqrt(percsidechain);
    }

    if ((pad_switch_env) > 0.0) {
        O += (pad_switch_env) * 1.5 * pad(mod(beat.z - 8.0, 64.0), 38.0) * sqrt(percsidechain);
    }

    // O = vec2(0.0);

    O += enable_sweep * sweep((beat.z - 16.0) * B2T) * 0.7;

    // FADEOUT
    // PAD + WAs
    // WAs trigger word
    // End WAs, play fnuque

    // TODO: n tweaking is good!!!
    if (enable_wah > 0.0) {
        float wah_beat = mod(beat.z - 4.0, 8.0);
        float wah_beat_num = floor((beat.z - 4.0) / 8.0);
        #define WAH(offset, note, pan, gain) if (wah_beat >= offset) { wahNote = note; wahNoteBeat = wah_beat - offset; wahPan = pan; wahGain = gain;  }
        float wahNote = 0.0;
        float wahNoteBeat = 0.0;
        float wahPan = 0.0;
        float wahGain = 0.0;

        if (block <= 3.0) {
            // repeat(n, 5)
            // {
            //     float note = 52.0;
            //     if (n == 3) {
            //         note = 55.0;
            //         if (pad_switch_beat < 4.0)
            //         {
            //             note = 57.0;
            //         }
            //         // On pitch down pad, use 57.0
            //     }

            //     float pan_ = 0.5;
            //     if (n == 0) {}
            //     else if (mod(n, 2) == 0) {
            //         pan_ = 0.1;
            //     }
            //     else if (mod(n, 2) == 1) {
            //         pan_ = 0.9;
            //     }
            //     O += enable_wah * 0.8 * pad3voice(mod(beat.z - 4.0 - n * 1.0, 8.0), floor((beat.z - 4.0) / 8.0), note) * exp(-n * 0.3) * pan(pan_, -4.5);
            // }

            float half_block = floor(beat.w / 32.0);
            float pad_switch_beat = mod(beat.z - 24.0, 32.0); // Last 2 bar in half block
            float pad_switch_env =
                (half_block >= 5.0 && half_block <= 7.0) ? linearenvwithhold(pad_switch_beat + 0.6, 0.6, 8.0, 0.6) : 0.0;

            float offset = 1.0;
            float offset_global_beat = beat.w + offset;
            float offset_half_block = offset_global_beat / 32.0;

            // float unfloored_half_block = offset_half_block

            bool modulated_pad = (offset_half_block >= 5.0 && offset_half_block < 8.0) && (mod(offset_half_block, 1.0) > 0.75);

            if (!modulated_pad) {
                WAH(0, 52, 0.5, exp(-0.0 * 0.3)) // E-2
                WAH(1, 52, 0.9, exp(-1.0 * 0.3))
                WAH(2, 52, 0.1, exp(-2.0 * 0.3))
                WAH(3, 55, 0.9, exp(-3.0 * 0.3)) // G-2
                WAH(4, 52, 0.1, exp(-4.0 * 0.3))
            } else {
                WAH(0, 50, 0.5, exp(-0.0 * 0.3)) // D-2
                WAH(1, 50, 0.9, exp(-1.0 * 0.3))
                WAH(2, 50, 0.1, exp(-2.0 * 0.3))
                WAH(3, 57, 0.9, exp(-3.0 * 0.3)) // A-2
                WAH(4, 50, 0.1, exp(-4.0 * 0.3))
            }
        } else {
            // Move to different rythm
            float wah_beat = mod(beat.z, 4.0);
            wah_beat_num *= 2.0;
            WAH(0, 52, 0.5, exp(-1.0 * 0.3))
            WAH(2.0, 52, 0.5, exp(-2.0 * 0.3))
            WAH(2.5, 55, 0.5, exp(-0.0 * 0.3))
        }
        O += 0.8 * pad3voice(wahNoteBeat, wah_beat_num, wahNote) * wahGain * pan(wahPan, -4.5);
    }
    // O += pad2voice(mod(beat.z, 8.0) - 1.5 - 0.125).yx * 0.5 * percsidechain;

    // O += mainbass_bar(barpos);
    // O += gnarly1bass_bar(barpos);
    // O += gnarly2bass_bar(barpos);
    // O += noisebass_bar(barpos);

    // O += riser2(mod(beat.z, 8.0) * B2T);

    // O += noisebass(time.y, p2f(40.0)) * 0.3;

    // O += shaker((beat.y-1.0)*B2T);
    // O += shaker((beat.y-2.0)*B2T);
    // O += shaker((beat.y-3.0)*B2T);

    // O += bassline(time, beat);

    return 1.0 * clamp(O, -1.0, 1.0);
}

void main() {
    int offset = int(gl_GlobalInvocationID.x) + waveOutPosition;
    float sec = float(offset) / SAMPLES_PER_SEC;
    waveOutSamples[offset] = 1. * mainSound(offset, sec);
}
