#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cctype>
#include <dirent.h>

// هدرهای هایکو
#include <SoundPlayer.h>
#include <MediaDefs.h>
#include <unistd.h>
#include <string.h>

// هدر پایپر
#include "piper_ncnn.h"

// ============================================================================
// کلاس‌های کمکی (قبل از کلاس اصلی تعریف می‌شوند)
// ============================================================================

// کلاسی برای ساخت و نوشتن داده‌های صوتی در یک فایل WAV
struct WavWriter {
    std::ofstream out;
    int samplerate, channels, bits;
    size_t data_bytes = 0;
    WavWriter(const std::string &path, int sr, int ch, int b) : samplerate(sr), channels(ch), bits(b) {
        out.open(path, std::ios::binary);
        if (!out) throw std::runtime_error("Could not open output file for writing: " + path);
        for (int i = 0; i < 44; i++) out.put(0);
    }
    void write(const std::vector<int16_t> &pcm) {
        out.write(reinterpret_cast<const char*>(pcm.data()), pcm.size() * sizeof(int16_t));
        data_bytes += pcm.size() * sizeof(int16_t);
    }
    void close() {
        out.seekp(0);
        int byte_rate = samplerate * channels * bits / 8;
        int block_align = channels * bits / 8;
        int subchunk2 = data_bytes;
        out.write("RIFF", 4); int chunk_size = 36 + subchunk2; out.write(reinterpret_cast<char*>(&chunk_size), 4);
        out.write("WAVE", 4); out.write("fmt ", 4); int subchunk1_size = 16; out.write(reinterpret_cast<char*>(&subchunk1_size), 4);
        short audio_format = 1; out.write(reinterpret_cast<char*>(&audio_format), 2); short ch = channels; out.write(reinterpret_cast<char*>(&ch), 2);
        int sr = samplerate; out.write(reinterpret_cast<char*>(&sr), 4); int br = byte_rate; out.write(reinterpret_cast<char*>(&br), 4);
        short ba = block_align; out.write(reinterpret_cast<char*>(&ba), 2); short bps = bits; out.write(reinterpret_cast<char*>(&bps), 2);
        out.write("data", 4); out.write(reinterpret_cast<char*>(&subchunk2), 4);
        out.close();
    }
};

// کلاسی که موتور Piper را مدیریت می‌کند
struct PiperEngine {
    piper_ncnn_synthesizer* synth = nullptr;
    ~PiperEngine() { if (synth) piper_ncnn_free(synth); }
    void init(const std::string& model_dir, const std::string& lang, int voice_id) {
        synth = piper_ncnn_create(model_dir.c_str(), lang.c_str(), voice_id);
        if (!synth) throw std::runtime_error("Failed to create Piper synthesizer.");
    }
    std::vector<int16_t> synthesize(const std::string& phoneme_sentence) {
        if (!synth || phoneme_sentence.empty()) return {};
        piper_ncnn_synthesize_options options = piper_ncnn_get_default_options(synth);
        if (piper_ncnn_synthesize_start(synth, phoneme_sentence.c_str(), &options) != PIPER_NCNN_OK) return {};
        std::vector<int16_t> pcm_data;
        piper_ncnn_audio_chunk chunk;
        while (piper_ncnn_synthesize_next(synth, &chunk) == PIPER_NCNN_OK) {
            if (chunk.num_samples > 0) pcm_data.insert(pcm_data.end(), chunk.samples, chunk.samples + chunk.num_samples);
            else break;
        }
        return pcm_data;
    }
};

// ============================================================================
// >> اصلاحیه ۱: اعلان پیشین برای تابع FillBuffer
// ============================================================================
void FillBuffer(void* cookie, void* buffer, size_t size, const media_raw_audio_format& format);


// ============================================================================
// کلاس اصلی برنامه
// ============================================================================

class HaikuttsApp {
public:
    HaikuttsApp(int argc, char** argv);
    int run();

private:
    // اعضای کلاس
    std::vector<std::string> m_args;
    std::string m_model_dir;
    int m_speaker_id;
    std::string m_text_arg;
    std::string m_infile;
    std::string m_outfile;
    std::string m_espeak_lang_override;

    std::string m_piper_lang;
    std::string m_espeak_voice;
    
    PiperEngine m_engine;
    std::queue<std::vector<int16_t>> m_audio_queue;
    std::mutex m_queue_mutex;

    // متدهای خصوصی
    void parse_arguments();
    void print_help();
    
    std::string find_model_name_from_dir(const std::string& model_dir);
    void parse_config(const std::string& path);
    std::string escape_for_single_quotes(const std::string& text);
    std::string get_phonemes(const std::string& lang, const std::string& text);
    std::vector<std::string> text_to_phoneme_sentences(const std::string& text);
    std::string normalize_persian_punctuation(const std::string& input);

    friend void FillBuffer(void* cookie, void* buffer, size_t size, const media_raw_audio_format& format);
};


// ============================================================================
// پیاده‌سازی متدهای کلاس HaikuttsApp
// ============================================================================

HaikuttsApp::HaikuttsApp(int argc, char** argv) : m_speaker_id(0) {
    if (argc > 0) {
        m_args.assign(argv, argv + argc);
    }
}

void HaikuttsApp::parse_arguments() {
    // ابتدا درخواست راهنما را بررسی کن
    for (size_t i = 1; i < m_args.size(); ++i) {
        if (m_args[i] == "-h" || m_args[i] == "--help") {
            print_help();
            exit(0);
        }
    }

    // متغیرهای موقت برای پردازش آرگومان‌ها
    std::string model_path = "/boot/home/config/non-packaged/data/HaikuTTS/models/"; // <<<< مسیر پیش‌فرض اصلاح شد
    std::string model_name = "";
    bool model_name_provided = false;
    
    // مقدار پیش‌فرض برای شناسه گوینده در سازنده کلاس تنظیم شده است

    // حلقه برای پردازش آپشن‌ها
    for (size_t i = 1; i < m_args.size(); ++i) {
        std::string arg = m_args[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < m_args.size()) {
            model_name = m_args[++i];
            model_name_provided = true;
        } else if ((arg == "-p" || arg == "--path") && i + 1 < m_args.size()) {
            model_path = m_args[++i];
        } else if ((arg == "-s" || arg == "--speaker_id") && i + 1 < m_args.size()) {
            m_speaker_id = std::stoi(m_args[++i]);
        } else if ((arg == "-l" || arg == "--lang") && i + 1 < m_args.size()) {
            m_espeak_lang_override = m_args[++i];
        } else if ((arg == "-t" || arg == "--text") && i + 1 < m_args.size()) {
            m_text_arg = m_args[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < m_args.size()) {
            m_outfile = m_args[++i];
        } else if ((arg == "-f" || arg == "--file") && i + 1 < m_args.size()) {
            m_infile = m_args[++i];
        }
    }

    // بررسی وجود آرگومان اجباری
    if (!model_name_provided) {
        throw std::runtime_error("Mandatory argument -m (--model) is missing. Use -h for help.");
    }

    // ساخت مسیر کامل دایرکتوری مدل
    if (!model_path.empty() && model_path.back() == '/') {
        m_model_dir = model_path + model_name;
    } else {
        m_model_dir = model_path + "/" + model_name;
    }

    // ادامه منطق تنظیمات برنامه بدون تغییر
    std::string model_name_base = find_model_name_from_dir(m_model_dir);
    std::string config_path = m_model_dir + "/" + model_name_base + "_config.txt";
    parse_config(config_path);
}


int HaikuttsApp::run() {
    parse_arguments();
    m_engine.init(m_model_dir, m_piper_lang, m_speaker_id);
    
    std::string final_espeak_voice = m_espeak_lang_override.empty() ? m_espeak_voice : m_espeak_lang_override;

    // آماده‌سازی پخش‌کننده یا فایل نویس
    WavWriter* writer = nullptr;
    BSoundPlayer* player = nullptr;
    if (!m_outfile.empty()) {
        writer = new WavWriter(m_outfile, 22050, 1, 16);
    } else {
        media_raw_audio_format format;
        memset(&format, 0, sizeof(format));
        format.frame_rate = 22050; format.channel_count = 1; format.format = media_raw_audio_format::B_AUDIO_SHORT;
        format.byte_order = B_MEDIA_LITTLE_ENDIAN; format.buffer_size = 4096;
        player = new BSoundPlayer(&format, "haikutts", FillBuffer, NULL, this);
        player->Start();
        player->SetHasData(true);
    }

    // ============================================================================
    // >> اصلاحیه ۲: جایگزینی unique_ptr با اشاره‌گر خام
    // ============================================================================
    std::istream* input_stream_ptr = nullptr;
    std::ifstream file_stream;
    std::stringstream text_stream;

    if (!m_text_arg.empty()) {
        text_stream.str(m_text_arg);
        input_stream_ptr = &text_stream;
    } else if (!m_infile.empty()) {
        file_stream.open(m_infile);
        if (!file_stream) throw std::runtime_error("Could not open input file: " + m_infile);
        input_stream_ptr = &file_stream;
    } else {
        input_stream_ptr = &std::cin;
    }

    // حلقه‌ی پردازشی واحد
    std::string line;
    while (std::getline(*input_stream_ptr, line)) {
        if (line.empty()) continue;
        
        std::string text_to_process = (m_piper_lang == "fa") ? normalize_persian_punctuation(line) : line;
        std::vector<std::string> phonemes_list = text_to_phoneme_sentences(text_to_process);

        for (const auto& p_sentence : phonemes_list) {
            auto pcm = m_engine.synthesize(p_sentence);
            if (writer) {
                writer->write(pcm);
            } else if (player && !pcm.empty()) {
                std::lock_guard<std::mutex> lock(m_queue_mutex);
                m_audio_queue.push(pcm);
            }
        }
    }
    if (file_stream.is_open()) file_stream.close();

    // پایان کار
    if (writer) {
        writer->close();
        delete writer;
    }
    if (player) {
        while (true) {
            bool queue_is_empty;
            { std::lock_guard<std::mutex> lock(m_queue_mutex); queue_is_empty = m_audio_queue.empty(); }
            if (queue_is_empty) { usleep(100000); break; }
            usleep(50000);
        }
        player->Stop();
        delete player;
    }
    return 0;
}


void HaikuttsApp::print_help() {
    std::cout << R"(
Usage: haikutts -m <model_name> [OPTIONS]

A command-line Text-to-Speech (TTS) program for the Haiku OS using Piper NCNN models.
The program automatically detects the model configuration from the specified directory.
Input text is read with the following priority: -t > -f > stdin.

## Mandatory Argument:
  -m, --model <model_name>  The name of the model directory to use.

## Options:
  -p, --path <path>         Path to the base directory containing model folders.
                            (Default: /boot/home/config/non-packaged/data/HaikuTTS/models/)

  -s, --speaker_id <id>     The integer ID of the speaker/voice to use.
                            (Default: 0).

  -l, --lang <lang_code>    Override the espeak voice used for phoneme generation
                            (e.g., "fr-fr"). If not provided, the "espeak_voice" value
                            from the config file is used.

  -t, --text "..."          The text string to synthesize. Highest priority input.

  -f, --file <path>         Read input text from a UTF-8 file.

  -o, --output <path>       Path to save the output as a WAV file. If omitted, the
                            synthesized audio will be played directly.

  -h, --help                Show this help message and exit.

--------------------------------------------------------------------------------

## Guidelines for Converting Piper ONNX Model

**References:**
* https://github.com/nihui/ncnn-android-piper
* https://github.com/OHF-Voice/piper1-gpl
* https://huggingface.co/datasets/rhasspy/piper-checkpoints

**Steps to convert Piper checkpoints to NCNN models:**

1.  **Checkout the correct version of the piper repository:**
    ```bash
    git clone [https://github.com/OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)
    cd piper1-gpl
    git checkout 113931937cf235fc8a11afd1ca4be209bc6919bc7
    ```

2.  **Apply the necessary patch:**
    ```bash
    # Ensure 'piper1-gpl.patch' is available
    git apply piper1-gpl.patch
    ```

3.  **Set up the Python environment and install dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -e .[train]
    ```

4.  **Download a Piper checkpoint file (`.ckpt`) from Hugging Face:**
    https://huggingface.co/datasets/rhasspy/piper-checkpoints

5.  **Install the PNNX model converter:**
    ```bash
    pip install -U pnnx
    ```

6.  **Obtain the `export_ncnn.py` script.**

7.  **Run the conversion script on your checkpoint file:**
    ```bash
    # Replace with your actual file
    python export_ncnn.py (language code).ckpt (e.g., en.ckpt, fa.ckpt, ...)
    ```

--------------------------------------------------------------------------------

**Created by:** gyroing (Amir Hossein Navabi)
* **Hugging Face:** https://huggingface.co/gyroing
* **GitHub:** https://github.com/gyroing
)" << std::endl;
}


std::string HaikuttsApp::find_model_name_from_dir(const std::string& model_dir) {
    DIR *dir = opendir(model_dir.c_str());
    if (!dir) throw std::runtime_error("Could not open model directory: " + model_dir);

    std::string model_name = "";
    int count = 0;
    const std::string suffix = "_config.txt";
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        std::string filename = ent->d_name;
        if (filename.length() > suffix.length() && filename.substr(filename.length() - suffix.length()) == suffix) {
            model_name = filename.substr(0, filename.length() - suffix.length());
            count++;
        }
    }
    closedir(dir);

    if (count == 0) throw std::runtime_error("No '_config.txt' file found in " + model_dir);
    if (count > 1) throw std::runtime_error("Multiple '_config.txt' files found. Directory must contain only one model.");
    return model_name;
}

void HaikuttsApp::parse_config(const std::string& path) {
    std::ifstream config_file(path);
    if (!config_file.is_open()) throw std::runtime_error("Cannot open config file: " + path);
    std::string line;
    while (std::getline(config_file, line)) {
        std::stringstream ss(line);
        std::string key, value;
        if (std::getline(ss, key, '=') && std::getline(ss, value)) {
            auto trim_string = [](const std::string& s, const std::string& delimiters = " \t\n\r\"") {
                size_t first = s.find_first_not_of(delimiters);
                if (std::string::npos == first) return s;
                size_t last = s.find_last_not_of(delimiters);
                return s.substr(first, (last - first + 1));
            };
            key = trim_string(key);
            value = trim_string(value);
            if (key == "espeak_voice") m_espeak_voice = value; 
            else if (key == "piper_language") m_piper_lang = value;
        }
    }
    if (m_espeak_voice.empty() || m_piper_lang.empty()) {
        throw std::runtime_error("'espeak_voice' or 'piper_language' not found in config file.");
    }
}

std::string HaikuttsApp::escape_for_single_quotes(const std::string& text) {
    std::string result;
    result.reserve(text.length());
    for (char c : text) {
        if (c == '\'') result.append("'\\''");
        else result.push_back(c);
    }
    return result;
}

std::string HaikuttsApp::get_phonemes(const std::string& lang, const std::string& text) {
    if (text.empty() || text.find_first_not_of(" \t\n\r") == std::string::npos) return "";
    std::string escaped_text = escape_for_single_quotes(text);
    std::string command = "espeak -q -v " + lang + " --ipa '" + escaped_text + "' 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) result += buffer;
    pclose(pipe);
    if (!result.empty() && result.back() == '\n') result.pop_back();
    return result;
}

std::vector<std::string> HaikuttsApp::text_to_phoneme_sentences(const std::string& text) {
    std::vector<std::string> phoneme_sentences;
    std::string final_espeak_voice = m_espeak_lang_override.empty() ? m_espeak_voice : m_espeak_lang_override;
    const std::string punct = ".,:;!?";
    size_t start_pos = 0;

    while (start_pos < text.length()) {
        size_t end_pos = text.find_first_of(punct, start_pos);
        std::string sentence_chunk;
        char punctuation_char = 0;

        if (end_pos == std::string::npos) {
            sentence_chunk = text.substr(start_pos);
            start_pos = text.length();
        } else {
            sentence_chunk = text.substr(start_pos, end_pos - start_pos);
            punctuation_char = text[end_pos];
            start_pos = end_pos + 1;
        }

        if (!sentence_chunk.empty() || punctuation_char != 0) {
            std::string chunk_phonemes = get_phonemes(final_espeak_voice, sentence_chunk);
            if (!chunk_phonemes.empty()) {
                 if (punctuation_char != 0) {
                    phoneme_sentences.push_back(chunk_phonemes + punctuation_char);
                 } else {
                    phoneme_sentences.push_back(chunk_phonemes);
                 }
            }
        }
    }
    return phoneme_sentences;
}


std::string HaikuttsApp::normalize_persian_punctuation(const std::string& input) {
    std::string output;
    output.reserve(input.length());
    for (size_t i = 0; i < input.length(); ++i) {
        const unsigned char c1 = static_cast<unsigned char>(input[i]);
        const unsigned char c2 = (i + 1 < input.length()) ? static_cast<unsigned char>(input[i+1]) : 0;
        if (c1 == 0xD8 && c2 == 0x9F) { output += '?'; i++; }
        else if (c1 == 0xD8 && c2 == 0x8C) { output += ','; i++; }
        else if (c1 == 0xD8 && c2 == 0x9B) { output += ';'; i++; }
        else { output += input[i]; }
    }
    return output;
}


// ============================================================================
// توابع سراسری
// ============================================================================

void FillBuffer(void* cookie, void* buffer, size_t size, const media_raw_audio_format& format) {
    HaikuttsApp* app = static_cast<HaikuttsApp*>(cookie);
    if (!app) return;

    std::lock_guard<std::mutex> lock(app->m_queue_mutex);
    size_t bytes_written = 0;
    while (bytes_written < size && !app->m_audio_queue.empty()) {
        auto& chunk = app->m_audio_queue.front();
        size_t chunk_bytes = chunk.size() * sizeof(int16_t);
        size_t bytes_to_copy = std::min(chunk_bytes, size - bytes_written);
        
        memcpy((char*)buffer + bytes_written, chunk.data(), bytes_to_copy);
        bytes_written += bytes_to_copy;

        if (bytes_to_copy == chunk_bytes) {
            app->m_audio_queue.pop();
        } else {
            auto& front_chunk = app->m_audio_queue.front();
            front_chunk.erase(front_chunk.begin(), front_chunk.begin() + (bytes_to_copy / sizeof(int16_t)));
        }
    }
    if (bytes_written < size) {
        memset((char*)buffer + bytes_written, 0, size - bytes_written);
    }
}


// ============================================================================
// تابع اصلی برنامه
// ============================================================================
int main(int argc, char** argv) {
    try {
        HaikuttsApp app(argc, argv);
        return app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
