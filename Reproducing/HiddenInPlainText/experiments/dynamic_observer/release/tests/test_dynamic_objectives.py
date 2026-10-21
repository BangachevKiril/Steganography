import unittest
from stego.dynamic_objectives import probabilities, sender_reward, schedule


class ObjectiveTests(unittest.TestCase):
    def test_common_logit_shift_invariance(self):
        self.assertEqual(probabilities([1., 2.]), probabilities([101., 102.]))

    def test_observer_uncertainty_optimal_and_label_symmetric(self):
        r=lambda p: sender_reward((.2,.8), (1-p,p), 1, 'x'*20)
        self.assertAlmostEqual(r(.5)['total'], .7)
        self.assertAlmostEqual(r(0.)['total'], -.3)
        self.assertEqual(r(0.),r(1.))
        self.assertAlmostEqual(r(.1)['total'],r(.9)['total'])

    def test_no_hard_gate_on_receiver(self):
        low=sender_reward((.8,.2),(0.,1.),1,'')
        high=sender_reward((.2,.8),(0.,1.),1,'')
        self.assertAlmostEqual(high['total']-low['total'],1.2)

    def test_character_not_token_bonus(self):
        for answer,expected in [('',0.),('a'*10,.05),('a'*20,.1),('a'*100,.1),('é'*10,.05)]:
            self.assertAlmostEqual(sender_reward((.5,.5),(.5,.5),0,answer)['character_bonus'],expected)

    def test_k_zero_is_frozen_label_observer(self):
        self.assertEqual(schedule(0),('receiver',)*2+('sender',)*5)
        self.assertEqual(sender_reward((.5,.5),(0.,1.),0,'')['observer'],-1.)
        for k in (1,2,4,8):
            self.assertEqual(schedule(k)[7:],('observer',)*k)

    def test_nonfinite_and_invalid_probabilities_fail(self):
        for p in [(float('nan'),.5),(-.1,1.1),(.2,.2)]:
            with self.assertRaises(ValueError):sender_reward(p,(.5,.5),0,'')
        with self.assertRaises(ValueError):probabilities([float('inf'),0.])


if __name__ == '__main__':unittest.main()
