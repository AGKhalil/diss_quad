import subprocess
import sys

if __name__ == "__main__":
    env_name = str(sys.argv[1])
    model_name = str(sys.argv[2])

    subprocess.Popen(
        '''export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so; python load_agent.py '%s' '%s' ''' % (env_name, model_name), shell=True)
