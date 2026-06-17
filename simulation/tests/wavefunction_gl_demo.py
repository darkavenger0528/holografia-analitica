import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyglet
import moderngl
import glm

from simulation.engine.wavefunction_realtime import (
    autoestados_oscilador,
    coef_coherente,
)

from simulation.renderer.renderer_gl import (
    WaveRendererGL,
)


class Window(pyglet.window.Window):

    def __init__(self):
        super().__init__(
            width=1280,
            height=720,
            caption="Wavefunction GPU",
            resizable=True,
        )

        self.ctx = moderngl.create_context()

        print("OpenGL:", self.ctx.info.get("GL_VERSION", "unknown"))
        print("GPU:", self.ctx.info.get("GL_RENDERER", "unknown"))

        self.renderer = WaveRendererGL(self.ctx)

        self.time = 0.0

        # IMPORTANTE: asignación correcta
        self.x, self.psis, self.E, _ = autoestados_oscilador()

        self.c = coef_coherente(
            2.5,
            len(self.E)
        )

        pyglet.clock.schedule_interval(
            self.update,
            1 / 60
        )

    def update(self, dt):
        self.time += dt

    def build_wave(self):

        phase = np.exp(
            -1j * self.E * self.time
        )

        psi = self.psis @ (
            self.c * phase
        )

        re = psi.real * 0.4
        im = psi.imag * 0.4

        density = np.abs(psi) ** 2
        density /= max(density.max(), 1e-9)

        hue = (
            np.angle(psi)
            /
            (2 * np.pi)
        ) % 1.0

        positions = np.column_stack([
            self.x,
            re,
            im,
        ])

        self.renderer.update(
            positions,
            hue,
            density,
        )

    def on_draw(self):

        self.clear()

        self.ctx.clear(
            0.0,
            0.0,
            0.0,
            1.0
        )

        self.build_wave()

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
            self.time * 0.5,
            glm.vec3(0, 1, 0)
        )

        mvp = proj * view * model

        self.renderer.render(mvp)


if __name__ == "__main__":

    try:
        window = Window()
        pyglet.app.run()

    except Exception:
        import traceback

        traceback.print_exc()

        input("\nPresiona ENTER para salir...")