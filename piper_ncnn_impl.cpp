#include "piper_ncnn.h"
#include "layer.h"
#include "mat.h"
#include "net.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <memory>
#include <cstring>

// ======================================================================
//          Custom Layers Implementation
// ======================================================================

class relative_embeddings_k_module : public ncnn::Layer
{
public:
    relative_embeddings_k_module() { one_blob_only = true; }
    
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        const int window_size = 4;
        const int wsize = bottom_blob.w;
        const int len = bottom_blob.h;
        const int num_heads = bottom_blob.c;
        
        top_blob.create(len, len, num_heads);
        top_blob.fill(0.f);
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++) {
            const ncnn::Mat x0 = bottom_blob.channel(q);
            ncnn::Mat out0 = top_blob.channel(q);
            
            for (int i = 0; i < len; i++) {
                const float* xptr = x0.row(i) + std::max(0, window_size - i);
                float* outptr = out0.row(i) + std::max(i - window_size, 0);
                
                const int wsize2 = std::min(len, i - window_size + wsize) - std::max(i - window_size, 0);
                for (int j = 0; j < wsize2; j++) {
                    *outptr++ = *xptr++;
                }
            }
        }
        return 0;
    }
};

class relative_embeddings_v_module : public ncnn::Layer
{
public:
    relative_embeddings_v_module() { one_blob_only = true; }
    
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        const int window_size = 4;
        const int wsize = window_size * 2 + 1;
        const int len = bottom_blob.h;
        const int num_heads = bottom_blob.c;
        
        top_blob.create(wsize, len, num_heads);
        top_blob.fill(0.f);
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++) {
            const ncnn::Mat x0 = bottom_blob.channel(q);
            ncnn::Mat out0 = top_blob.channel(q);
            
            for (int i = 0; i < len; i++) {
                const float* xptr = x0.row(i) + std::max(i - window_size, 0);
                float* outptr = out0.row(i) + std::max(0, window_size - i);
                
                const int wsize2 = std::min(len, i - window_size + wsize) - std::max(i - window_size, 0);
                for (int j = 0; j < wsize2; j++) {
                    *outptr++ = *xptr++;
                }
            }
        }
        return 0;
    }
};

class piecewise_rational_quadratic_transform_module : public ncnn::Layer
{
public:
    piecewise_rational_quadratic_transform_module() { one_blob_only = false; }
    
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& h = bottom_blobs[0];
        const ncnn::Mat& x1 = bottom_blobs[1];
        ncnn::Mat& outputs = top_blobs[0];
        
        const int num_bins = 10;
        const int filter_channels = 192;
        const bool reverse = true;
        const float tail_bound = 5.0f;
        const float DEFAULT_MIN_BIN_WIDTH = 1e-3f;
        const float DEFAULT_MIN_BIN_HEIGHT = 1e-3f;
        const float DEFAULT_MIN_DERIVATIVE = 1e-3f;
        
        const int batch_size = x1.w;
        outputs = x1.clone();
        float* out_ptr = outputs;
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < batch_size; ++i) {
            const float current_x = ((const float*)x1)[i];
            const float* h_data = h.row(i);
            
            if (current_x < -tail_bound || current_x > tail_bound) {
                continue;
            }
            
            std::vector<float> unnormalized_widths(num_bins);
            std::vector<float> unnormalized_heights(num_bins);
            std::vector<float> unnormalized_derivatives(num_bins + 1);
            
            const float inv_sqrt_filter_channels = 1.0f / sqrtf(filter_channels);
            
            for (int j = 0; j < num_bins; ++j) {
                unnormalized_widths[j] = h_data[j] * inv_sqrt_filter_channels;
            }
            
            for (int j = 0; j < num_bins; ++j) {
                unnormalized_heights[j] = h_data[num_bins + j] * inv_sqrt_filter_channels;
            }
            
            for (int j = 0; j < num_bins - 1; ++j) {
                unnormalized_derivatives[j + 1] = h_data[2 * num_bins + j];
            }
            
            const float constant = logf(expf(1.f - DEFAULT_MIN_DERIVATIVE) - 1.f);
            unnormalized_derivatives[0] = constant;
            unnormalized_derivatives[num_bins] = constant;
            
            const float left = -tail_bound;
            const float right = tail_bound;
            const float bottom = -tail_bound;
            const float top = tail_bound;
            
            std::vector<float> widths(num_bins);
            float w_max = -INFINITY;
            for (float val : unnormalized_widths) {
                w_max = std::max(w_max, val);
            }
            
            float w_sum = 0.f;
            for (int j = 0; j < num_bins; ++j) {
                widths[j] = expf(unnormalized_widths[j] - w_max);
                w_sum += widths[j];
            }
            
            for (int j = 0; j < num_bins; ++j) {
                widths[j] = DEFAULT_MIN_BIN_WIDTH + (1.f - DEFAULT_MIN_BIN_WIDTH * num_bins) * (widths[j] / w_sum);
            }
            
            std::vector<float> cumwidths(num_bins + 1);
            cumwidths[0] = left;
            float current_w_sum = 0.f;
            
            for (int j = 0; j < num_bins - 1; ++j) {
                current_w_sum += widths[j];
                cumwidths[j + 1] = left + (right - left) * current_w_sum;
            }
            cumwidths[num_bins] = right;
            
            std::vector<float> heights(num_bins);
            float h_max = -INFINITY;
            for (float val : unnormalized_heights) {
                h_max = std::max(h_max, val);
            }
            
            float h_sum = 0.f;
            for (int j = 0; j < num_bins; ++j) {
                heights[j] = expf(unnormalized_heights[j] - h_max);
                h_sum += heights[j];
            }
            
            for (int j = 0; j < num_bins; ++j) {
                heights[j] = DEFAULT_MIN_BIN_HEIGHT + (1.f - DEFAULT_MIN_BIN_HEIGHT * num_bins) * (heights[j] / h_sum);
            }
            
            std::vector<float> cumheights(num_bins + 1);
            cumheights[0] = bottom;
            float current_h_sum = 0.f;
            
            for (int j = 0; j < num_bins - 1; ++j) {
                current_h_sum += heights[j];
                cumheights[j + 1] = bottom + (top - bottom) * current_h_sum;
            }
            cumheights[num_bins] = top;
            
            std::vector<float> derivatives(num_bins + 1);
            for (int j = 0; j < num_bins + 1; ++j) {
                float x = unnormalized_derivatives[j];
                derivatives[j] = DEFAULT_MIN_DERIVATIVE + (x > 0 ? x + logf(1.f + expf(-x)) : logf(1.f + expf(x)));
            }
            
            int bin_idx = 0;
            if (reverse) {
                auto it = std::upper_bound(cumheights.begin(), cumheights.end(), current_x);
                bin_idx = std::distance(cumheights.begin(), it) - 1;
            } else {
                auto it = std::upper_bound(cumwidths.begin(), cumwidths.end(), current_x);
                bin_idx = std::distance(cumwidths.begin(), it) - 1;
            }
            
            bin_idx = std::max(0, std::min(bin_idx, num_bins - 1));
            
            const float input_cumwidths = cumwidths[bin_idx];
            const float input_bin_widths = cumwidths[bin_idx + 1] - cumwidths[bin_idx];
            const float input_cumheights = cumheights[bin_idx];
            const float input_heights = cumheights[bin_idx + 1] - cumheights[bin_idx];
            const float input_derivatives = derivatives[bin_idx];
            const float input_derivatives_plus_one = derivatives[bin_idx + 1];
            const float delta = input_heights / input_bin_widths;
            
            if (reverse) {
                float a = (current_x - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * delta) + input_heights * (delta - input_derivatives);
                float b = input_heights * input_derivatives - (current_x - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * delta);
                float c = -delta * (current_x - input_cumheights);
                float discriminant = b * b - 4 * a * c;
                discriminant = std::max(0.f, discriminant);
                float root = (2 * c) / (-b - sqrtf(discriminant));
                out_ptr[i] = root * input_bin_widths + input_cumwidths;
            } else {
                float theta = (current_x - input_cumwidths) / input_bin_widths;
                float theta_one_minus_theta = theta * (1 - theta);
                float numerator = input_heights * (delta * theta * theta + input_derivatives * theta_one_minus_theta);
                float denominator = delta + ((input_derivatives + input_derivatives_plus_one - 2 * delta) * theta_one_minus_theta);
                out_ptr[i] = input_cumheights + numerator / denominator;
            }
        }
        return 0;
    }
};

// ======================================================================
//          Custom Layer Creator Wrappers
// ======================================================================

static ncnn::Layer* relative_embeddings_k_module_creator(void* /*userdata*/) {
    return new relative_embeddings_k_module();
}

static ncnn::Layer* relative_embeddings_v_module_creator(void* /*userdata*/) {
    return new relative_embeddings_v_module();
}

static ncnn::Layer* piecewise_rational_quadratic_transform_module_creator(void* /*userdata*/) {
    return new piecewise_rational_quadratic_transform_module();
}

// ======================================================================
//          Internal Structures & Helper Functions
// ======================================================================

struct PiperNCNNInternal {
    std::string model_dir;
    std::string language;
    std::map<std::string, int> phoneme_map;
    std::unique_ptr<ncnn::Mat> speaker_embedding;
    bool is_multi_speaker;
    int sample_rate;
    
    float noise_scale;
    float length_scale;
    float noise_scale_w;
    int num_speakers;
    int current_speaker_id;

    ncnn::Net enc_p;
    ncnn::Net dp;
    ncnn::Net flow;
    ncnn::Net dec;

    // Internal buffer to hold the synthesized audio
    std::vector<int16_t> internal_audio_buffer;
    size_t current_buffer_pos;
    
    PiperNCNNInternal() : 
        is_multi_speaker(false),
        sample_rate(22050),
        noise_scale(0.667f),
        length_scale(1.0f),
        noise_scale_w(0.8f),
        num_speakers(1),
        current_speaker_id(0),
        current_buffer_pos(0) {}
};

static bool load_config_internal(const std::string& path, PiperNCNNInternal* internal) {
    std::ifstream infile(path);
    if (!infile.is_open()) {
        return false;
    }
    
    internal->num_speakers = 1;
    
    std::string line;
    while (std::getline(infile, line)) {
        size_t separator_pos = line.find('=');
        if (separator_pos == std::string::npos) continue;
        
        std::string key_str = line.substr(0, separator_pos);
        std::string value_str = line.substr(separator_pos + 1);
        
        key_str.erase(0, key_str.find_first_not_of(" \t\n\r"));
        key_str.erase(key_str.find_last_not_of(" \t\n\r") + 1);
        value_str.erase(0, value_str.find_first_not_of(" \t\n\r"));
        value_str.erase(value_str.find_last_not_of(" \t\n\r") + 1);
        
        if (key_str.size() >= 2 && key_str.front() == '"' && key_str.back() == '"') {
            key_str = key_str.substr(1, key_str.size() - 2);
        }
        if (value_str.size() >= 2 && value_str.front() == '"' && value_str.back() == '"') {
            value_str = value_str.substr(1, value_str.size() - 2);
        }
        
        try {
            if (key_str == "noise_scale") {
                internal->noise_scale = std::stof(value_str);
            } else if (key_str == "length_scale") {
                internal->length_scale = std::stof(value_str);
            } else if (key_str == "noise_w") {
                internal->noise_scale_w = std::stof(value_str);
            } else if (key_str == "num_speakers") {
                internal->num_speakers = std::stoi(value_str);
            } else if (key_str == "sample_rate") {
                internal->sample_rate = std::stoi(value_str);
            }
        } catch (const std::exception& e) {
            // Silently ignore
        }
    }
    return true;
}

static int load_phoneme_map_internal(const std::string& path, std::map<std::string, int>& phoneme_map) {
    phoneme_map.clear();
    std::ifstream infile(path);
    if (!infile.is_open()) {
        return -1;
    }
    
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        
        size_t tab_pos = line.find('\t');
        if (tab_pos == std::string::npos) {
            tab_pos = line.find(' ');
            if (tab_pos == std::string::npos) continue;
        }
        
        std::string phoneme_quoted = line.substr(0, tab_pos);
        std::string id_str = line.substr(tab_pos + 1);
        
        phoneme_quoted.erase(0, phoneme_quoted.find_first_not_of(" \t\n\r"));
        phoneme_quoted.erase(phoneme_quoted.find_last_not_of(" \t\n\r") + 1);
        id_str.erase(0, id_str.find_first_not_of(" \t\n\r"));
        id_str.erase(id_str.find_last_not_of(" \t\n\r") + 1);
        
        if (phoneme_quoted.size() >= 2 && phoneme_quoted.front() == '"' && phoneme_quoted.back() == '"') {
            phoneme_quoted = phoneme_quoted.substr(1, phoneme_quoted.size() - 2);
        }
        
        try {
            int id = std::stoi(id_str);
            phoneme_map[phoneme_quoted] = id;
        } catch (const std::exception& e) {
            // Silently ignore
        }
    }
    return 0;
}

static int get_char_width(const char* pchar) {
    unsigned char c = ((const unsigned char*)pchar)[0];
    if (c < 128) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

static void phonemes_to_ids_internal(const std::string& phoneme_text, 
                                   const std::map<std::string, int>& phoneme_map,
                                   std::vector<int>& phoneme_ids) {
    phoneme_ids.clear();
    
    if (phoneme_map.empty() || 
        phoneme_map.find("^") == phoneme_map.end() ||
        phoneme_map.find("_") == phoneme_map.end() || 
        phoneme_map.find("$") == phoneme_map.end()) {
        return;
    }
    
    phoneme_ids.push_back(phoneme_map.at("^"));
    phoneme_ids.push_back(phoneme_map.at("_"));
    
    const char* p = phoneme_text.c_str();
    const char* end = p + phoneme_text.length();
    
    while (p < end && *p) {
        int char_len = get_char_width(p);
        std::string phoneme_char(p, char_len);
        
        if (phoneme_map.find(phoneme_char) != phoneme_map.end()) {
            phoneme_ids.push_back(phoneme_map.at(phoneme_char));
            phoneme_ids.push_back(phoneme_map.at("_"));
        } else if (phoneme_char != " " && phoneme_char != "\n" && phoneme_char != "\r" && phoneme_char != "\t") {
            // Skip unknown
        }
        p += char_len;
    }
    
    phoneme_ids.push_back(phoneme_map.at("$"));
}

static void path_attention(const ncnn::Mat& logw, const ncnn::Mat& m_p, const ncnn::Mat& logs_p, 
                          float noise_scale, float length_scale, ncnn::Mat& z_p) {
    const int x_lengths = logw.w;
    const int depth = m_p.h;
    
    std::vector<int> w_ceil(x_lengths);
    int y_lengths = 0;
    
    for (int i = 0; i < x_lengths; i++) {
        w_ceil[i] = (int)ceilf(expf(logw[i]) * length_scale);
        y_lengths += w_ceil[i];
    }
    
    z_p.create(y_lengths, depth);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < depth; i++) {
        const float* m_p_ptr = m_p.row(i);
        const float* logs_p_ptr = logs_p.row(i);
        float* ptr = z_p.row(i);
        
        for (int j = 0; j < x_lengths; j++) {
            const float m = m_p_ptr[j];
            const float logs = logs_p_ptr[j];
            const float nl = expf(logs) * noise_scale;
            const int duration = w_ceil[j];
            
            for (int k = 0; k < duration; k++) {
                ptr[k] = m + dis(gen) * nl;
            }
            ptr += duration;
        }
    }
}

// ======================================================================
//          C Interface Implementation
// ======================================================================

piper_ncnn_synthesize_options piper_ncnn_get_default_options(piper_ncnn_synthesizer *synth) {
    piper_ncnn_synthesize_options options;
    if (synth) {
        PiperNCNNInternal* internal = reinterpret_cast<PiperNCNNInternal*>(synth);
        options.length_scale = internal->length_scale;
        options.noise_scale = internal->noise_scale;
        options.noise_w_scale = internal->noise_scale_w;
    } else {
        options.length_scale = 1.0f;
        options.noise_scale = 0.667f;
        options.noise_w_scale = 0.8f;
    }
    return options;
}

piper_ncnn_synthesizer *piper_ncnn_create(const char *model_dir, const char* language, int speaker_id) {
    if (!model_dir || !language) {
        return nullptr;
    }
    
    auto internal = std::make_unique<PiperNCNNInternal>();
    internal->model_dir = model_dir;
    internal->language = language;
    internal->current_speaker_id = speaker_id;
    
    std::string config_path = std::string(model_dir) + "/" + internal->language + "_config.txt";
    if (!load_config_internal(config_path.c_str(), internal.get())) {
        std::cerr << "Error: Failed to load config file: " << config_path << std::endl;
        return nullptr;
    }
    
    if (internal->current_speaker_id < 0 || internal->current_speaker_id >= internal->num_speakers) {
        internal->current_speaker_id = 0;
    }
    
    std::string phoneme_map_path = std::string(model_dir) + "/" + internal->language + "_phoneme_id_map.txt";
    if (load_phoneme_map_internal(phoneme_map_path, internal->phoneme_map) != 0) {
        return nullptr;
    }
    
    if (internal->num_speakers > 1) {
        ncnn::Net emb_g;
        emb_g.opt.use_vulkan_compute = false;
        std::string emb_param_path = std::string(model_dir) + "/" + internal->language + "_emb_g.ncnn.param";
        std::string emb_bin_path = std::string(model_dir) + "/" + internal->language + "_emb_g.ncnn.bin";
        
        if (emb_g.load_param(emb_param_path.c_str()) == 0 && emb_g.load_model(emb_bin_path.c_str()) == 0) {
            internal->is_multi_speaker = true;
            ncnn::Mat speaker_id_mat(1);
            ((int*)speaker_id_mat.data)[0] = internal->current_speaker_id;
            
            ncnn::Extractor ex = emb_g.create_extractor();
            ex.input("in0", speaker_id_mat);
            ncnn::Mat g;
            if (ex.extract("out0", g) == 0) {
                internal->speaker_embedding.reset(new ncnn::Mat(g.reshape(1, g.w)));
            }
        }
    }
    
    const std::string& lang = internal->language;
    const std::string& m_dir = internal->model_dir;

    internal->enc_p.opt.use_vulkan_compute = false;
    internal->dp.opt.use_vulkan_compute = false;
    internal->flow.opt.use_vulkan_compute = false;
    internal->dec.opt.use_vulkan_compute = false;

    internal->enc_p.register_custom_layer("piper.train.vits.attentions.relative_embeddings_k_module", relative_embeddings_k_module_creator);
    internal->enc_p.register_custom_layer("piper.train.vits.attentions.relative_embeddings_v_module", relative_embeddings_v_module_creator);
    internal->dp.register_custom_layer("piper.train.vits.modules.piecewise_rational_quadratic_transform_module", piecewise_rational_quadratic_transform_module_creator);

    if (internal->enc_p.load_param((m_dir + "/" + lang + "_enc_p.ncnn.param").c_str()) != 0 ||
        internal->enc_p.load_model((m_dir + "/" + lang + "_enc_p.ncnn.bin").c_str()) != 0) return nullptr;

    if (internal->dp.load_param((m_dir + "/" + lang + "_dp.ncnn.param").c_str()) != 0 ||
        internal->dp.load_model((m_dir + "/" + lang + "_dp.ncnn.bin").c_str()) != 0) return nullptr;

    if (internal->flow.load_param((m_dir + "/" + lang + "_flow.ncnn.param").c_str()) != 0 ||
        internal->flow.load_model((m_dir + "/" + lang + "_flow.ncnn.bin").c_str()) != 0) return nullptr;
    
    if (internal->dec.load_param((m_dir + "/" + lang + "_dec.ncnn.param").c_str()) != 0 ||
        internal->dec.load_model((m_dir + "/" + lang + "_dec.ncnn.bin").c_str()) != 0) return nullptr;
    
    return reinterpret_cast<piper_ncnn_synthesizer*>(internal.release());
}

void piper_ncnn_free(piper_ncnn_synthesizer *synth) {
    if (synth) {
        delete reinterpret_cast<PiperNCNNInternal*>(synth);
    }
}

int piper_ncnn_synthesize_start(piper_ncnn_synthesizer *synth, const char *phonemes, const piper_ncnn_synthesize_options *options) {
    if (!synth || !phonemes) {
        return PIPER_NCNN_ERR_INVALID_INPUT;
    }
    
    PiperNCNNInternal* internal = reinterpret_cast<PiperNCNNInternal*>(synth);
    
    internal->internal_audio_buffer.clear();
    internal->current_buffer_pos = 0;

    float current_noise_scale = internal->noise_scale;
    float current_length_scale = internal->length_scale;
    float current_noise_w_scale = internal->noise_scale_w;

    if (options) {
        current_noise_scale = options->noise_scale;
        current_length_scale = options->length_scale;
        current_noise_w_scale = options->noise_w_scale;
    }
    
    try {
        std::vector<int> sequence_ids;
        phonemes_to_ids_internal(phonemes, internal->phoneme_map, sequence_ids);
        
        if (sequence_ids.size() <= 3) {
            return PIPER_NCNN_ERR_INVALID_INPUT;
        }
        
        ncnn::Mat sequence((int)sequence_ids.size());
        for (size_t i = 0; i < sequence_ids.size(); i++) {
            ((int*)sequence.data)[i] = sequence_ids[i];
        }
        
        ncnn::Mat x, m_p, logs_p;
        ncnn::Extractor ex_enc = internal->enc_p.create_extractor();
        ex_enc.input("in0", sequence);
        if (ex_enc.extract("out0", x) != 0) return PIPER_NCNN_ERR_SYNTHESIS_FAILED;
        if (ex_enc.extract("out1", m_p) != 0) return PIPER_NCNN_ERR_SYNTHESIS_FAILED;
        if (ex_enc.extract("out2", logs_p) != 0) return PIPER_NCNN_ERR_SYNTHESIS_FAILED;
        
        ncnn::Mat noise(x.w, 2);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < noise.w * noise.h; i++) {
            noise[i] = dis(gen) * current_noise_w_scale;
        }
        
        ncnn::Mat logw;
        ncnn::Extractor ex_dp = internal->dp.create_extractor();
        ex_dp.input("in0", x);
        ex_dp.input("in1", noise);
        if (internal->is_multi_speaker && internal->speaker_embedding) {
            ex_dp.input("in2", *internal->speaker_embedding);
        }
        if (ex_dp.extract("out0", logw) != 0) return PIPER_NCNN_ERR_SYNTHESIS_FAILED;
        
        ncnn::Mat z_p;
        path_attention(logw, m_p, logs_p, current_noise_scale, current_length_scale, z_p);
        
        ncnn::Mat z;
        ncnn::Extractor ex_flow = internal->flow.create_extractor();
        ex_flow.input("in0", z_p);
        if (internal->is_multi_speaker && internal->speaker_embedding) {
            ex_flow.input("in1", *internal->speaker_embedding);
        }
        if (ex_flow.extract("out0", z) != 0) return PIPER_NCNN_ERR_SYNTHESIS_FAILED;
        
        ncnn::Mat o;
        ncnn::Extractor ex_dec = internal->dec.create_extractor();
        ex_dec.input("in0", z);
        if (internal->is_multi_speaker && internal->speaker_embedding) {
            ex_dec.input("in1", *internal->speaker_embedding);
        }
        if (ex_dec.extract("out0", o) != 0) return PIPER_NCNN_ERR_SYNTHESIS_FAILED;
        
        float absmax = 0.f;
        for (int i = 0; i < o.w; i++) {
            absmax = std::max(absmax, fabs(o[i]));
        }
        if (absmax > 1e-8f) {
            for (int i = 0; i < o.w; i++) {
                o[i] = std::min(std::max(o[i] / absmax, -1.f), 1.f);
            }
        }
        
        internal->internal_audio_buffer.resize(o.w);
        for (int i = 0; i < o.w; i++) {
            internal->internal_audio_buffer[i] = static_cast<int16_t>(o[i] * 32767.f);
        }
        
        return PIPER_NCNN_OK;
        
    } catch (const std::exception& e) {
        return PIPER_NCNN_ERR_GENERIC;
    }
}

int piper_ncnn_synthesize_next(piper_ncnn_synthesizer *synth, piper_ncnn_audio_chunk *chunk) {
    if (!synth || !chunk) {
        return PIPER_NCNN_ERR_INVALID_INPUT;
    }
    
    PiperNCNNInternal* internal = reinterpret_cast<PiperNCNNInternal*>(synth);
    
    if (internal->current_buffer_pos >= internal->internal_audio_buffer.size()) {
        chunk->samples = nullptr;
        chunk->num_samples = 0;
        return PIPER_NCNN_DONE;
    }
    
    const size_t chunk_size = 4096;
    size_t remaining_samples = internal->internal_audio_buffer.size() - internal->current_buffer_pos;
    size_t samples_to_return = std::min(chunk_size, remaining_samples);
    
    chunk->samples = &internal->internal_audio_buffer[internal->current_buffer_pos];
    chunk->num_samples = samples_to_return;
    
    internal->current_buffer_pos += samples_to_return;
    
    return PIPER_NCNN_OK;
}

int piper_ncnn_synthesize_line(piper_ncnn_synthesizer *synth, const char *phonemes, const piper_ncnn_synthesize_options *options) {
    int result = piper_ncnn_synthesize_start(synth, phonemes, options);
    if (result != PIPER_NCNN_OK) {
        return result;
    }

    piper_ncnn_audio_chunk chunk;
    while (piper_ncnn_synthesize_next(synth, &chunk) == PIPER_NCNN_OK) {
        if (chunk.num_samples > 0) {
            fwrite(chunk.samples, sizeof(piper_ncnn_audio_sample), chunk.num_samples, stdout);
        }
    }
    
    fflush(stdout);
    return PIPER_NCNN_OK;
}

const char* piper_ncnn_strerror(int error_code) {
    switch (error_code) {
        case PIPER_NCNN_OK: return "Success";
        case PIPER_NCNN_DONE: return "Operation done";
        case PIPER_NCNN_ERR_GENERIC: return "Generic error";
        case PIPER_NCNN_ERR_INVALID_INPUT: return "Invalid input provided";
        case PIPER_NCNN_ERR_MODEL_LOAD_FAILED: return "Failed to load one or more models";
        case PIPER_NCNN_ERR_SYNTHESIS_FAILED: return "Synthesis failed during processing";
        default: return "Unknown error";
    }
}

