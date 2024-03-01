#pragma once

#include <deque>
#include <chrono>
#include <string>
#include <iostream>

struct PerfCounters {

    using time_point = std::chrono::high_resolution_clock::time_point;
    using time_point_pair = std::pair<time_point, time_point>;

    struct Sample {
        // Next frame total
        std::chrono::microseconds total = {};
        std::chrono::microseconds decode = {};
        std::chrono::microseconds processing = {};
        // Vaapi to USM
        std::chrono::microseconds va2usm = {};
        std::chrono::microseconds va2usm_sync = {};
        // Workaround
        std::chrono::microseconds wa_mem = {};

        Sample& operator+=(const Sample& rhs) {
            total += rhs.total;
            decode += rhs.decode;
            processing += rhs.processing;
            va2usm += rhs.va2usm;
            va2usm_sync += rhs.va2usm_sync;
            wa_mem += rhs.wa_mem;
            return *this;
        }

        Sample& operator/=(size_t divider) {
            total /= divider;
            decode /= divider;
            processing /= divider;
            va2usm /= divider;
            va2usm_sync /= divider;
            wa_mem /= divider;
            return *this;
        }
    };
    std::deque<Sample> samples;
    Sample active_sample;

    static std::chrono::microseconds get_duration(time_point_pair tp_pair) {
        using namespace std::chrono;
        return duration_cast<microseconds>(tp_pair.second - tp_pair.first);
    }

    void sample_track_vappi_to_usm(time_point_pair total, time_point_pair sync) {
        active_sample.va2usm = get_duration(total);
        active_sample.va2usm_sync = get_duration(sync);
    }

    void sample_track_wa_memory(time_point_pair total) {
        active_sample.wa_mem = get_duration(total);
    }

    void sample_track_decode(time_point_pair total) {
        active_sample.decode = get_duration(total);
    }

    void sample_add_processing(time_point_pair total) {
        active_sample.processing += get_duration(total);
    }

    void complete_sample(time_point_pair total, time_point_pair decode,
                         time_point_pair processing) {
        active_sample.decode = get_duration(decode);
        active_sample.processing = get_duration(processing);

        complete_sample(total);
    }

    void complete_sample(time_point_pair total) {
        active_sample.total = get_duration(total);

        samples.emplace_back(active_sample);
        active_sample = Sample();
    }

    static void print_sample(const Sample& s, std::string_view header = {}) {
        if (header.empty())
            header = "Performace sample:";

        std::cout << header << std::endl
                  << "  Total:        " << s.total.count() << "us\n"
                  << "  - Decode:     " << s.decode.count() << "us\n"
                  << "  - Processing: " << s.processing.count() << "us\n"
                  << "    * VA2USM:   " << s.va2usm.count() << "us\n"
                  << "      > sync:   " << s.va2usm_sync.count() << "us\n"
                  << "    * Mem W/A:  " << s.wa_mem.count() << "us\n";
    }

    void print_last() const {
        const auto& s = samples.back();
        const std::string header = "Perf sample[" + std::to_string(samples.size() - 1) + "]:";
        print_sample(s, header);
    }

    void print_all() const {
        size_t cnt = 0;
        for (const auto& s : samples) {
            const std::string header = "Perf sample[" + std::to_string(cnt++) + "]:";
            print_sample(s, header);
        }
    }

    void print_avg(size_t skip_n = 0) const {
        if (skip_n > samples.size()) {
            std::cout << "Cannot print AVG: skip size is larger than number of samples\n";
            return;
        }

        Sample avg;
        for (auto it = samples.cbegin() + skip_n; it != samples.cend(); ++it)
            avg += *it;

        avg /= samples.size() - skip_n;
        const std::string header = "Perf samples[" + std::to_string(skip_n) + ':' +
                                   std::to_string(samples.size() - 1) + "] AVG:";
        print_sample(avg, header);
    }
};
