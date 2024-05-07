class Frame:
    def __init__(self, c_frame, stream_id, frame_id) -> None:
        self.__c_obj = c_frame[0]
        self.__stream_id = stream_id
        self.__frame_id = frame_id
        if len(c_frame) > 1:
            self.__original_frame = c_frame[1]

    @property
    def va_display(self):
        return self.__c_obj.va_display

    @property
    def va_surface_id(self):
        return self.__c_obj.va_surface_id

    @property
    def stream_id(self):
        return self.__stream_id

    @property
    def frame_id(self):
        return self.__frame_id

    @property
    def width(self):
        return self.__c_obj.width

    @property
    def height(self):
        return self.__c_obj.height

    @property
    def raw(self):
        return self.__c_obj

    @property
    def original_frame(self):
        return self.__original_frame
