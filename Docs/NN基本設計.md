# 学習の流れ

1. データの読み込み  
画像配置フォルダに判定したい画像のサンプルを配置
画像ファイル名は「タイプ_ナンバー.拡張子」とする。

1. データの調整  
保存した画像データを学習のために最適化する

    1. データサイズの均一化
    1. データの水増し
        1. 位置の変化
        1. 向きの変化
        1. 明度の変化  
        ※ これらは、画像データの仕様によって調整
    
    1. 学習用フォーマットに合わせてオブジェクト化して保存

1. ネットワークの構築

1. 学習

# ツールの流れ

1. 画像の保存  
判定対象フォルダに画像を保存

1. 判定プログラム実行
    1. 画像の読み込み
        1. 画像をフォーマットに合わせて変換
        1. 判定用フォルダにデータを保存
    1. 判定プログラム実行
        1. 学習したネットワークの読み込み
        1. 画像の判定
        1. 判定結果の出力
            1. 画像表示機能を使用するといいかも  
            pyplotlibとか

# 機能概要

## 画像データ変換機能

用意したデータを自動で学習できるフォーマットに変換する

## 学習フレームワーク

フォーマットされたデータを読み込んで
精度が確保できるまで半自動で学習する

学習用パラメータを自動で設定したり、
簡単に調整できるような仕組みを。

## ツール

判定したい画像を入れて起動すると、
判定結果が出力されるプログラム。

# 将来的な仕様

+ スマホからとったデータのpcへの転送
+ 転送された時点で判定プログラムの実行