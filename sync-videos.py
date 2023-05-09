"""
Automatically resynchronize video files
"""
import cv2 as cv
import numpy as np

class OpticalFlowComputer:
    def __init__(self, fn, args):
        self.cap = cv.VideoCapture(fn)
        self.args = args

        self.frame_no = 0
        self.src = self._next()
        self.hsv = None

    def next_scalar_flow(self):
        target = self._next()
        if target is None: return None
        self.flow = cv.calcOpticalFlowFarneback(self.src, target, None, 0.5, 3, self.args.flow_winsize, 3, 5, 1.2, 0)
        #flow_scalar = np.mean(np.mean(self.flow, axis=0), axis=0)

        stride = self.args.flow_winsize
        flow_scalar = np.ravel(self.flow[::stride, ::stride, ...])

        self.src = target
        return flow_scalar

    def _next(self):
        while True:
            self.frame_no += 1
            frame = self._grab_frame()

            if frame is None: return None
            # print('frame %d' % frame_no)
            if self.frame_no <= args.skip_first_n_frames: continue
            
            return self._convert_frame(frame)

    def _grab_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None
        return frame
    
    def _convert_frame(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if self.args.resize_width > 0:
            h, w = frame.shape
            w1 = self.args.resize_width
            h1 = h * w1 // w
            frame = cv.resize(frame, (w1, h1), interpolation=cv.INTER_AREA)
        return frame
    
    def show_preview(self, wnd_tag=''):
        if not self.args.preview: return

        cv.imshow('input ' + wnd_tag, self.src)

        if self.hsv is None:
            self.hsv = np.zeros((self.src.shape[0], self.src.shape[1], 3), dtype=self.src.dtype)
            self.hsv[..., 1] = 255
        
        mag, ang = cv.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        self.hsv[..., 0] = ang*180/np.pi/2
        self.hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(self.hsv, cv.COLOR_HSV2BGR)
        cv.imshow('optical flow ' + wnd_tag, bgr)
        cv.waitKey(1)

def minimize_error(flow1, flow2, max_offset):

    result_x = np.zeros(max_offset*2+1, dtype=int)
    result_y = np.zeros(max_offset*2+1, dtype=float)

    def shift(offset):
        if offset >= 0:
            shifted1 = flow1[offset:,:]
            shifted2 = flow2[:shifted1.shape[0],:]
        else:
            shifted2 = flow2[(-offset):,:]
            shifted1 = flow1[:shifted2.shape[0],:]
        return shifted1, shifted2

    for i in range(len(result_x)):
        offset = i - max_offset
        result_x[i] = offset

        shifted1, shifted2 = shift(offset)
        result_y[i] = np.mean(np.abs(shifted1 - shifted2))

        # plt.plot(np.mean(shifted1, axis=1))
        # plt.plot(np.mean(shifted2, axis=1))
        # plt.show()
        # print(result_y[i])

    offset = int(result_x[np.argmin(result_y)])
    print(offset)

    return offset, (result_x, result_y), shift(offset)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('leader_video')
    p.add_argument('follower_video')
    p.add_argument('leader_data_jsonl', nargs='?')
    p.add_argument('--flow_winsize', type=int, default=15)
    p.add_argument('-skip', '--skip_first_n_frames', type=int, default=0, help='Skip first N frames')
    p.add_argument('--preview', action='store_true')
    p.add_argument('--no_plot', action='store_true')
    p.add_argument('--flip_second_flow', action='store_true')
    p.add_argument('--ffmpeg_flags', default='-y -an -c:v libx264 -crf 18')
    p.add_argument('--resize_width', type=int, default=200)
    p.add_argument('--max_frames', type=int, default=1000)
    p.add_argument('--max_offset', type=int, default=100)
    p.add_argument('--dry_run', action='store_true')

    args = p.parse_args()
    leader_flow = OpticalFlowComputer(args.leader_video, args)
    follower_flow = OpticalFlowComputer(args.follower_video, args)

    flow1 = []
    flow2 = [] 

    for i in range(args.max_frames):
        f1 = leader_flow.next_scalar_flow()
        f2 = follower_flow.next_scalar_flow()
        if f1 is None or f2 is None: break
        if args.flip_second_flow: f2 = -f2
        flow1.append(list(f1))
        flow2.append(list(f2))
        leader_flow.show_preview('leader')
        follower_flow.show_preview('follower')
        
    flow1 = np.array(flow1)
    flow2 = np.array(flow2)

    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.plot(np.mean(flow1, axis=1), label='leader (mean)')
        plt.plot(np.mean(flow2, axis=1), label='follower (mean)')
        plt.legend()
        plt.show()

    optimal_offset, (plot_x, plot_y), (opt1, opt2) = minimize_error(flow1, flow2, args.max_offset)

    if not args.no_plot:
        plt.plot(plot_x, plot_y)
        plt.title('error by offset (optimum %d)' % optimal_offset)
        plt.show()

        plt.plot(np.mean(opt1, axis=1), label='leader, shifted')
        plt.plot(np.mean(opt2, axis=1), label='follower, shifted')
        plt.show()

    def sync_command_ffmpeg(fn, skip_n_frames, out_fn=None):
        if out_fn is None: out_fn = fn + '.resynced.' + fn.rpartition('.')[-1]
        if skip_n_frames == 0:
            return 'cp %s %s' % (fn, out_fn)
        else:
            return "ffmpeg -i %s -vf 'select=gte(n\,%d)' %s %s" % (fn, skip_n_frames, args.ffmpeg_flags, out_fn)

    skip1 = args.skip_first_n_frames + max(optimal_offset, 0)
    skip2 = args.skip_first_n_frames + max(-optimal_offset, 0)

    sync1 = sync_command_ffmpeg(args.leader_video, skip1)
    sync2 = sync_command_ffmpeg(args.follower_video, skip2)
    print(sync1)
    print(sync2)

    if not args.dry_run:
        import json, os, threading
        if args.leader_data_jsonl is not None:
            with open(args.leader_data_jsonl, 'rt') as f_in:
                with open(args.leader_data_jsonl + '.synced.jsonl', 'wt') as f_out:
                    for line in f_in:
                        d = json.loads(line)
                        if 'frames' in d:
                            if d['number'] < skip1: continue
                            d['number'] -= skip1
                        f_out.write(json.dumps(d)+'\n')

        t1 = threading.Thread(target=lambda: os.system(sync1))
        t2 = threading.Thread(target=lambda: os.system(sync2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    cv.destroyAllWindows()