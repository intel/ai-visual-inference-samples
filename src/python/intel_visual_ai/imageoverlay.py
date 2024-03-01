from PIL import Image, ImageDraw, ImageFont


class ImageOverlay:
    def __init__(self, scale=5):
        self.scale = scale
        self.font = ImageFont.load_default()

    def draw_text(self, image, text, position=(10, 10), color=(255, 0, 0)):
        """
        Draw magnified text on an image.

        Args:
            image (PIL.Image): The image to draw on.
            text (str): The text to draw.
            position (tuple): The position to draw the text (x, y).
            color (tuple): The color of the text.

        Returns:
            PIL.Image: The image with text drawn on it.
        """
        draw = ImageDraw.Draw(image)

        # Measure the size of the text using textbbox
        bbox = draw.textbbox(position, text, font=self.font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Create a temporary image for the text
        temp_image = Image.new("RGBA", (text_width, text_height), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_image)
        temp_draw.text((0, 0), text, font=self.font, fill=color)

        # Resize the temporary image to magnify the text
        text_image = temp_image.resize(
            (text_width * self.scale, text_height * self.scale), Image.NEAREST
        )

        # Paste the magnified text back onto the original image
        image.paste(text_image, position, text_image)

        return image
