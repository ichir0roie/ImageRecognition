"""

主な機能のうち、学習を行う。
mainSetImageを実行したあと、こちらを実行すると、用意した画像に対して学習を実行する。
学習完了後はmainPredictを実行することで、用意した画像の判定を行えるようになる。

やることはLearn.pyを実行するだけ
"""

import MyPackageNetwork.Learner

learner=MyPackageNetwork.Learner.Learner()

learner.run()