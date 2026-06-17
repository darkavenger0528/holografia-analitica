from pathlib import Path

import moderngl
import numpy as np


class WaveRendererGL:

    def __init__(self, ctx):

        self.ctx = ctx

        shader_dir = (
            Path(__file__).parent
            / "shaders"
        )

        with open(shader_dir / "wave.vert") as f:
            vertex_shader = f.read()

        with open(shader_dir / "wave.frag") as f:
            fragment_shader = f.read()

        self.program = ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        self.vbo = None
        self.vao = None

    def update(
        self,
        positions,
        phases,
        densities
    ):

        data = np.column_stack([
            positions,
            phases,
            densities
        ]).astype("f4")

        if self.vbo is not None:
            self.vbo.release()

        self.vbo = self.ctx.buffer(
            data.tobytes()
        )

        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (
                    self.vbo,
                    "3f 1f 1f",
                    "in_position",
                    "in_phase",
                    "in_density"
                )
            ]
        )

    def render(self, mvp):

        self.program["mvp"].write(
            np.array(
                mvp,
                dtype="f4"
            ).tobytes()
        )

        self.vao.render(
            mode=moderngl.POINTS
        )