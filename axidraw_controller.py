import time
from pyaxidraw import axidraw


class AxiDrawController:

    def __init__(
        self,
        port="COM5",
        model=2,
        canvas_width_in=16.54,
        canvas_height_in=11.69,
        margin_in=0.9,
        right_safety_in=0.4,
        bottom_safety_in=0.7,
        home=(0.0, 0.0)
    ):
        self.port = port
        self.model = model
        self.canvas_width = canvas_width_in
        self.canvas_height = canvas_height_in
        self.margin = margin_in
        self.right_safety = right_safety_in
        self.bottom_safety = bottom_safety_in
        self.home = home

        self.ad = axidraw.AxiDraw()
        self.connected = False

    def connect(self):
        self.ad.interactive()
        self.ad.options.port = self.port
        self.ad.options.model = self.model
        self.ad.options.pen_pos_up = 100
        self.ad.options.pen_pos_down = 40
        self.ad.options.pen_delay_up = 400
        self.ad.options.pen_delay_down = 400
        self.ad.options.speed_pendown = 25
        self.ad.options.speed_penup = 45

        if not self.ad.connect():
            print("Failed to connect to AxiDraw.")
            return False

        self.ad.update()
        self.connected = True
        self.pen_up()

        print(
            "AxiDraw connected.",
            "| model:", self.model,
            "| canvas:", f"{self.canvas_width} x {self.canvas_height} in",
            "| base margin:", self.margin,
            "| right safety:", self.right_safety,
            "| bottom safety:", self.bottom_safety,
            "| pen up:", self.ad.options.pen_pos_up,
            "| pen down:", self.ad.options.pen_pos_down
        )

        return True

    def disconnect(self, return_home=True):
        if not self.connected:
            return

        self.pen_up()

        if return_home:
            self.go_home()

        self.ad.disconnect()
        self.connected = False
        print("AxiDraw disconnected.")

    def pen_up(self):
        if not self.connected:
            return
        self.ad.penup()
        time.sleep(0.6)

    def pen_down(self):
        if not self.connected:
            return
        self.ad.pendown()
        time.sleep(0.6)

    def go_home(self):
        if not self.connected:
            return

        self.pen_up()
        self.ad.moveto(self.home[0], self.home[1])
        self.ad.block()
        time.sleep(0.5)

    def draw_line(self, start, end, return_home=False):
        if not self.connected:
            return

        safe_start = self._clamp_canvas_point(start)
        safe_end = self._clamp_canvas_point(end)

        self.pen_up()
        self.ad.moveto(safe_start[0], safe_start[1])
        self.ad.block()
        time.sleep(0.3)

        self.pen_down()
        self.ad.lineto(safe_end[0], safe_end[1])
        self.ad.block()
        self.pen_up()

        if return_home:
            self.go_home()

    def draw_polyline(
        self,
        points,
        return_home=False,
        segment_delay=0.05
    ):
        if not self.connected:
            return

        if points is None or len(points) < 2:
            return

        safe_points = [
            self._clamp_canvas_point(point)
            for point in points
        ]

        first_point = safe_points[0]

        self.pen_up()
        self.ad.moveto(first_point[0], first_point[1])
        self.ad.block()
        time.sleep(0.3)

        self.pen_down()

        for point in safe_points[1:]:
            self.ad.lineto(point[0], point[1])
            time.sleep(segment_delay)

        self.ad.block()
        self.pen_up()

        if return_home:
            self.go_home()

    def draw_action(
        self,
        action,
        frame_shape,
        return_home=False
    ):
        if not self.connected or action is None:
            return

        action_type = action.get("type")

        if action_type == "definition_line":
            if "start" not in action or "end" not in action:
                return

            start = self._map_point(action["start"], frame_shape)
            end = self._map_point(action["end"], frame_shape)

            print("Camera line:", action["start"], action["end"])
            print("AxiDraw line:", start, end)

            self.draw_line(
                start=start,
                end=end,
                return_home=return_home
            )
            return

        if action_type == "definition_polyline":
            digital_points = action.get("points")

            if digital_points is None or len(digital_points) < 2:
                return

            physical_points = [
                self._map_point(point, frame_shape)
                for point in digital_points
            ]

            print("Camera polyline:", digital_points)
            print("AxiDraw polyline:", physical_points)

            self.draw_polyline(
                points=physical_points,
                return_home=return_home,
                segment_delay=action.get("segment_delay", 0.05)
            )

    def _map_point(self, point, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        x_px, y_px = point

        x_px = max(0, min(float(x_px), frame_width))
        y_px = max(0, min(float(y_px), frame_height))

        min_x = self.margin
        max_x = (
            self.canvas_width
            - self.margin
            - self.right_safety
        )

        min_y = self.margin
        max_y = (
            self.canvas_height
            - self.margin
            - self.bottom_safety
        )

        drawable_width = max_x - min_x
        drawable_height = max_y - min_y

        x_in = min_x + (x_px / frame_width) * drawable_width
        y_in = min_y + (y_px / frame_height) * drawable_height

        return self._clamp_canvas_point((x_in, y_in))

    def _clamp_canvas_point(self, point):
        x, y = point

        min_x = self.margin
        max_x = (
            self.canvas_width
            - self.margin
            - self.right_safety
        )

        min_y = self.margin
        max_y = (
            self.canvas_height
            - self.margin
            - self.bottom_safety
        )

        safe_x = max(min_x, min(float(x), max_x))
        safe_y = max(min_y, min(float(y), max_y))

        return safe_x, safe_y