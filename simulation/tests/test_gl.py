import pyglet
import moderngl
import numpy as np
import glm


class TestWindow(pyglet.window.Window):

    def __init__(self):
        super().__init__(
            width=1280,
            height=720,
            caption="ModernGL Test",
            resizable=True
        )

        self.ctx = moderngl.create_context()

        print("OpenGL:")
        print(self.ctx.info["GL_VENDOR"])
        print(self.ctx.info["GL_RENDERER"])
        print(self.ctx.info["GL_VERSION"])

        n = 10000

        positions = np.random.uniform(
            -1.0,
            1.0,
            (n, 3)
        ).astype("f4")

        colors = np.random.uniform(
            0.2,
            1.0,
            (n, 3)
        ).astype("f4")

        data = np.hstack(
            [positions, colors]
        )

        self.vbo = self.ctx.buffer(
            data.tobytes()
        )

        self.program = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;
                in vec3 in_color;

                out vec3 v_color;

                uniform mat4 mvp;

                void main()
                {
                    gl_Position = mvp * vec4(
                        in_position,
                        1.0
                    );

                    gl_PointSize = 4.0;

                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330

                in vec3 v_color;

                out vec4 fragColor;

                void main()
                {
                    vec2 p =
                        gl_PointCoord -
                        vec2(0.5);

                    if(length(p) > 0.5)
                        discard;

                    fragColor =
                        vec4(v_color, 1.0);
                }
            """
        )

        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (
                    self.vbo,
                    "3f 3f",
                    "in_position",
                    "in_color"
                )
            ]
        )

        self.time = 0.0

    def on_draw(self):

        self.clear()

        self.ctx.clear(
            0.0,
            0.0,
            0.0
        )

        self.time += 0.01

        proj = glm.perspective(
            glm.radians(45.0),
            self.width / self.height,
            0.1,
            100.0
        )

        view = glm.lookAt(
            glm.vec3(0, 0, 4),
            glm.vec3(0, 0, 0),
            glm.vec3(0, 1, 0)
        )

        model = glm.rotate(
            glm.mat4(1.0),
            self.time,
            glm.vec3(0, 1, 0)
        )

        mvp = proj * view * model

        self.program["mvp"].write(
            np.array(mvp, dtype="f4").tobytes()
        )

        self.vao.render(
            moderngl.POINTS
        )


if __name__ == "__main__":
    window = TestWindow()
    pyglet.app.run()