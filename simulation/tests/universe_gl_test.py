import math
import numpy as np
import pyglet
import moderngl
import glm

WINDOW_W = 1280
WINDOW_H = 720

N_PARTICLES = 100000

VERTEX_SHADER = """
#version 330

in vec3 in_pos;
in vec3 in_color;

out vec3 v_color;

uniform float scale_factor;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    vec3 pos = in_pos * scale_factor;

    gl_Position =
        projection *
        view *
        model *
        vec4(pos, 1.0);

    gl_PointSize = 3.0;

    v_color = in_color;
}
"""

FRAGMENT_SHADER = """
#version 330

in vec3 v_color;
out vec4 fragColor;

void main()
{
    float d = length(gl_PointCoord - vec2(0.5));

    if(d > 0.5)
        discard;

    fragColor = vec4(v_color, 1.0);
}
"""


class UniverseGLWindow(pyglet.window.Window):

    def __init__(self):

        super().__init__(
            width=WINDOW_W,
            height=WINDOW_H,
            caption="Universe OpenGL Test",
            resizable=True
        )

        self.ctx = moderngl.create_context()

        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        rng = np.random.default_rng(42)

        pos = rng.uniform(
            -1.0,
            1.0,
            (N_PARTICLES, 3)
        ).astype("f4")

        colors = rng.uniform(
            0.2,
            1.0,
            (N_PARTICLES, 3)
        ).astype("f4")

        vertices = np.hstack([
            pos,
            colors
        ]).astype("f4")

        self.vbo = self.ctx.buffer(vertices.tobytes())

        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (
                    self.vbo,
                    "3f 3f",
                    "in_pos",
                    "in_color"
                )
            ]
        )

        self.scale_factor = 0.01
        self.rotation = 0.0

        pyglet.clock.schedule_interval(
            self.update,
            1 / 60
        )

    def update(self, dt):

        self.rotation += dt * 0.25

        self.scale_factor += dt * 0.02

        if self.scale_factor > 1.0:
            self.scale_factor = 0.01

    def on_draw(self):

        self.clear()

        self.ctx.clear(0, 0, 0)

        projection = glm.perspective(
            glm.radians(45.0),
            self.width / self.height,
            0.1,
            100.0
        )

        view = glm.lookAt(
            glm.vec3(0, 0, 3),
            glm.vec3(0, 0, 0),
            glm.vec3(0, 1, 0)
        )

        model = glm.rotate(
            glm.mat4(1.0),
            self.rotation,
            glm.vec3(0, 1, 0)
        )

        self.program["projection"].write(
            np.ascontiguousarray(
                np.array(projection, dtype="f4")
            )
        )

        self.program["view"].write(
            np.ascontiguousarray(
                np.array(view, dtype="f4")
            )
        )

        self.program["model"].write(
            np.ascontiguousarray(
                np.array(model, dtype="f4")
            )
        )

        self.program["scale_factor"].value = (
            self.scale_factor
        )

        self.vao.render(
            moderngl.POINTS
        )


if __name__ == "__main__":

    UniverseGLWindow()

    pyglet.app.run()