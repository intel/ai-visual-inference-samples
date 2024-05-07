#pragma once

#include <pybind11/pybind11.h>

#ifdef ENABLE_PYLOGGER

// Simple logger implementaion based on Python logging module
// IMPORTANT:
//   Does NOT thread-safe!
//   Calls to this object imply that GIL is being held!
class Logger final {
  private:
    pybind11::object pylogger_;
    pybind11::object log_;
    pybind11::object is_level_enabled_;

    enum class Level : int {
        trace = 0,
        debug = 10,
        info = 20,
        warn = 30,
        error = 40,
        critical = 50,
    };

    template <class... Args>
    void log(Level level, std::string_view fmt, Args&&... args) const {
        using namespace pybind11;

        // Test if message should be logged
        const int level_int = static_cast<int>(level);
        object should_log = is_level_enabled_(level_int);
        if (!isinstance<bool_>(should_log))
            return;
        if (!cast<bool>(should_log))
            return;

        auto formatted_str = str(fmt).format(std::forward<Args>(args)...);
        log_(level_int, std::move(formatted_str));
    }

  public:
    Logger() : Logger("visual_ai") {}
    Logger(std::string_view logger_name) {
        auto get_logger = pybind11::module::import("logging").attr("getLogger");
        pylogger_ = get_logger(logger_name);
        log_ = pylogger_.attr("log");
        is_level_enabled_ = pylogger_.attr("isEnabledFor");
    }

    template <class... Args>
    void debug(std::string_view fmt, Args&&... args) const {
        log(Level::debug, fmt, std::forward<Args>(args)...);
    }

    template <class... Args>
    void info(std::string_view fmt, Args&&... args) const {
        log(Level::info, fmt, std::forward<Args>(args)...);
    }

    template <class... Args>
    void warn(std::string_view fmt, Args&&... args) const {
        log(Level::warn, fmt, std::forward<Args>(args)...);
    }

    template <class... Args>
    void error(std::string_view fmt, Args&&... args) const {
        log(Level::error, fmt, std::forward<Args>(args)...);
    }

    template <class... Args>
    void critical(std::string_view fmt, Args&&... args) const {
        log(Level::critical, fmt, std::forward<Args>(args)...);
    }
};

#else

// Dummy implementation
class Logger final {
  public:
    Logger() = default;
    Logger(std::string_view /*logger_name*/) {}

    template <class... Args>
    void debug(std::string_view /*fmt*/, Args&&... /*args*/) const {}

    template <class... Args>
    void info(std::string_view /*fmt*/, Args&&... /*args*/) const {}

    template <class... Args>
    void warn(std::string_view /*fmt*/, Args&&... /*args*/) const {}

    template <class... Args>
    void error(std::string_view /*fmt*/, Args&&... /*args*/) const {}

    template <class... Args>
    void critical(std::string_view /*fmt*/, Args&&... /*args*/) const {}
};

#endif // ENABLE_PYLOGGER