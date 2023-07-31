import numpy as np

def make_gif(images, fname, duration = 2, fps = 24):
    """ generate gif animation
        
        Parameters
        ----------
        images : array of image [frame, width, height, channels]
        
        Returns
        -------
        no
        """
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        return x.astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=fps, verbose=False, opt=None)
        
        
def make_video(images, fname, duration = 2, fps = 24):
    """ generate video animation
        this function use the same way of .gif just change file name to .mp4 or .avi
        see https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html for detail
        
        Parameters
        ----------
        images : array of image [frame, width, height, channels]
        
        Returns
        -------
        no
        """
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        return x.astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_videofile(fname, fps=fps)
