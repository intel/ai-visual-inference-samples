#pragma once

// Abstract base class for frame
class Frame {
  protected:
    Frame() = default;

  public:
    virtual ~Frame() = default;

    // NO COPY
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
};
