"""CPU integration checks in the legacy training environment."""
import copy,json,random,tempfile,unittest
from pathlib import Path
import torch
from stego.university_training import ShuffleStream
from stego.dynamic_training import ObserverClient
from stego.university_judge import validate_config
from orcd.qwen_observer_client import text_hashes

class Preflight(unittest.TestCase):
    def test_receiver_wrap_and_resume(self):
        torch.manual_seed(42)
        stream=ShuffleStream(range(9000),256)
        seen=[]
        for _ in range(36):
            batch=stream.next();self.assertEqual(len(batch),256);seen.extend(batch)
        self.assertEqual(sorted(seen[:9000]),list(range(9000)))
        saved=copy.deepcopy(stream.state_dict());rng=torch.get_rng_state()
        expected=stream.next()
        other=ShuffleStream(range(9000),256);other.load_state_dict(saved)
        torch.set_rng_state(rng);self.assertEqual(other.next(),expected)
    def test_sender_drop_last(self):
        stream=ShuffleStream(range(9000),192,True)
        for _ in range(46):self.assertEqual(len(stream.next()),192)
        self.assertEqual(stream.epoch,1)
        self.assertEqual(len(stream.next()),192);self.assertEqual(stream.epoch,2)
    def test_observer_corrupt_response_rejected(self):
        client=object.__new__(ObserverClient)
        response=dict(count=1,text_sha256=text_hashes(['x']),rows=[])
        client._request=lambda *a,**kw:response
        with self.assertRaises(ValueError):client.score(['x'])
        response['rows']=[dict(observer_probabilities=[.2,.8],observer_prediction=1,observer_confidence=.8)]
        self.assertEqual(client.score(['x']),response['rows'])
        response['rows'][0]['observer_confidence']=.4
        with self.assertRaises(ValueError):client.score(['x'])
    def test_all_configs_and_equal_cycles(self):
        root=Path(__file__).resolve().parents[1]
        for p in (root/'configs').glob('dynamic_*.json'):
            c=json.loads(p.read_text());validate_config(c)
            from stego.dynamic_objectives import schedule
            phases=schedule(c['dynamic']['k'])*c['dynamic']['cycles']
            self.assertEqual(phases.count('sender'),1000)
            self.assertEqual(phases.count('receiver'),400)
            self.assertEqual(phases.count('observer'),200*c['dynamic']['k'])

if __name__=='__main__':unittest.main()
