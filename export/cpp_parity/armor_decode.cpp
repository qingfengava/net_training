// Runs sleepy's real ArmorInfer decoder (AT mode) on a raw network output and prints what it makes of it.
//
//   armor_decode OUTPUT.f32 [ROWS] [CONF_THRESHOLD]   ->  JSON on stdout
//
// OUTPUT.f32 is ROWS x 18 float32, row-major, the layout nt/export.py writes: box4, conf, number, 4 color
// scores, 8 keypoint coordinates. This file includes sleepy's headers and links its built libraries from the
// outside; nothing in the sleepy checkout is changed. Build and run it with scripts/run_cpp_parity.sh.
#include "tasks/auto_aim/armor_detect/armor_infer.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

using namespace sleepy::auto_aim;

static const char* color_name(ArmorColor c) {
    switch (c) {
        case ArmorColor::BLUE: return "blue";
        case ArmorColor::RED: return "red";
        case ArmorColor::NONE: return "none";
        case ArmorColor::PURPLE: return "purple";
    }
    return "?";
}

static const char* class_name(ArmorClass c) {
    switch (c) {
        case ArmorClass::SENTRY: return "sentry";
        case ArmorClass::NO1: return "one";
        case ArmorClass::NO2: return "two";
        case ArmorClass::NO3: return "three";
        case ArmorClass::NO4: return "four";
        case ArmorClass::NO5: return "five";
        case ArmorClass::OUTPOST: return "outpost";
        case ArmorClass::BASE: return "base";
        case ArmorClass::UNKNOWN: return "unknown";
    }
    return "?";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: armor_decode OUTPUT.f32 [ROWS=30] [CONF=0.2]\n";
        return 2;
    }
    const int rows = argc > 2 ? std::atoi(argv[2]) : 30;
    const double conf = argc > 3 ? std::atof(argv[3]) : 0.2;
    constexpr int kCols = 18;

    std::vector<float> data(static_cast<size_t>(rows) * kCols);
    std::ifstream in(argv[1], std::ios::binary);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (static_cast<size_t>(in.gcount()) != data.size() * sizeof(float)) {
        std::cerr << "expected " << data.size() * sizeof(float) << " bytes, got " << in.gcount() << "\n";
        return 3;
    }

    YAML::Node cfg;
    cfg["model_type"] = "at2";
    cfg["conf_threshold"] = conf;
    ArmorInfer infer(cfg);

    cv::Mat out(rows, kCols, CV_32F, data.data());
    const auto armors = infer.process(out);

    std::printf("{\"input_w\":%d,\"input_h\":%d,\"use_norm\":%s,\"armors\":[", infer.input_w(), infer.input_h(),
                infer.use_norm() ? "true" : "false");
    bool first = true;
    for (const auto& a: armors) {
        if (!a.net) continue;
        const auto& p = a.net->key_points.points;
        auto pt = [&](ArmorKeyPointsIndex i) {
            const auto& v = p[std::to_underlying(i)];
            std::printf("[%.6f,%.6f]", v ? v->x : -1.f, v ? v->y : -1.f);
        };
        std::printf("%s{\"conf\":%.6f,\"number\":\"%s\",\"color\":\"%s\",\"lt\":", first ? "" : ",", a.net->confidence,
                    class_name(a.net->number), color_name(a.net->color));
        pt(ArmorKeyPointsIndex::LEFT_TOP);
        std::printf(",\"lb\":");
        pt(ArmorKeyPointsIndex::LEFT_BOTTOM);
        std::printf(",\"rb\":");
        pt(ArmorKeyPointsIndex::RIGHT_BOTTOM);
        std::printf(",\"rt\":");
        pt(ArmorKeyPointsIndex::RIGHT_TOP);
        std::printf("}");
        first = false;
    }
    std::printf("]}\n");
    return 0;
}
