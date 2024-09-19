import sys
import os
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from client.client import SocketIOClient
from client.agent import CustomAgentBase

class MyAgent(CustomAgentBase):
    def __init__(self):
        super().__init__()
        self.card_model = self.load_or_train_card_model()
        self.koikoi_model = self.load_or_train_koikoi_model()
        self.card_data = []
        self.koikoi_data = []

    def load_or_train_card_model(self):
        if os.path.exists('card_model.pkl'):
            with open('card_model.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            return self.train_card_model()

    def load_or_train_koikoi_model(self):
        if os.path.exists('koikoi_model.pkl'):
            with open('koikoi_model.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            return self.train_koikoi_model()

    def train_card_model(self):
        X, y = self.load_card_data()
        if len(X) == 0:
            X, y = self.generate_card_training_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def train_koikoi_model(self):
        X, y = self.load_koikoi_data()
        if len(X) == 0:
            X, y = self.generate_koikoi_training_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def load_card_data(self):
        if os.path.exists('card_data.pkl'):
            with open('card_data.pkl', 'rb') as f:
                data = pickle.load(f)
            return np.array([d[0] for d in data]), np.array([d[1] for d in data])
        return [], []

    def load_koikoi_data(self):
        if os.path.exists('koikoi_data.pkl'):
            with open('koikoi_data.pkl', 'rb') as f:
                data = pickle.load(f)
            return np.array([d[0] for d in data]), np.array([d[1] for d in data])
        return [], []

    def save_card_data(self):
        with open('card_data.pkl', 'wb') as f:
            pickle.dump(self.card_data, f)

    def save_koikoi_data(self):
        with open('koikoi_data.pkl', 'wb') as f:
            pickle.dump(self.koikoi_data, f)

    def save_models(self):
        with open('card_model.pkl', 'wb') as f:
            pickle.dump(self.card_model, f)
        with open('koikoi_model.pkl', 'wb') as f:
            pickle.dump(self.koikoi_model, f)

    def generate_card_training_data(self):
        # カード選択のための訓練データを生成
        X = []
        y = []
        for _ in range(100000):  # 10,000サンプル生成
            hand = [random.randint(1, 12) for _ in range(8)]
            field = [random.randint(1, 12) for _ in range(8)]
            features = hand + field
            action = random.choice(hand)
            X.append(features)
            y.append(action)
        return np.array(X), np.array(y)

    def generate_koikoi_training_data(self):
        # こいこい決定のための訓練データを生成
        X = []
        y = []
        for _ in range(10000):  # 10,000サンプル生成
            your_yaku = random.randint(0, 5)
            your_score = random.randint(0, 30)
            op_score = random.randint(0, 30)
            features = [your_yaku, your_score, op_score]
            action = random.choice([0, 1])  # 0: しない, 1: する
            X.append(features)
            y.append(action)
        return np.array(X), np.array(y)

    def select_card(self, hand, field):
        features = hand + field
        prediction = self.card_model.predict([features])[0]
        if prediction in hand:
            self.card_data.append((features, prediction))
            return prediction
        else:
            chosen_card = self.find_matching_card(hand, field)
            self.card_data.append((features, chosen_card))
            return chosen_card

    def decide_koikoi(self, your_yaku, your_score, op_score):
        features = [len(your_yaku), your_score, op_score]
        prediction = self.koikoi_model.predict([features])[0]
        self.koikoi_data.append((features, prediction))
        return bool(prediction)

    def find_matching_card(self, hand_cards, fields):
        """手札のカードと一致する場札を探す"""
        print('find_card関数が呼び出されました')
        same_month = []
        for hand_card in hand_cards:
            for field_card in fields:
                if hand_card[0] == field_card[0]:
                    same_month.append(field_card)
        print('一致する場札:', same_month)
        
        if len(same_month) == 1:
            print('一致する場札が1枚見つかりました:', same_month[0])
            return same_month[0]
        
        elif len(same_month) == 0:
            # 一致する場札がない場合、ランダムで選ぶ
            chosen_card = random.choice(hand_cards)
            print('一致する場札がないため、ランダムで選ばれたカード:', chosen_card)
            return chosen_card
        else:
            # 一致する場札が複数ある場合、点数の高いカードを選ぶ
            same_month.sort(key=lambda x: x[1])
            print('一致する場札が複数あるため、点数の高いカードを選びました:', same_month[0])
            return same_month[0]

    def custom_act(self, observation):
        """盤面情報と取れる行動を受け取って，行動を決定して返す関数．参加者が各自で実装．"""
        hand = observation['your_hand']
        field = observation['field']
        your_yaku = observation['your_yaku']
        your_score = observation['your_total_point']
        op_score = observation['op_total_point']
        game_state = observation['state']
        legal_actions = observation['legal_action']

        print('--------data----------')
        print('hand', hand)
        print('field', field)
        print('your_yaku', your_yaku)
        print('your_score', your_score)
        print('op_score', op_score)
        print('game_state', game_state)
        print('legal_actions', legal_actions)
        print('----------------------')
        
        if game_state == 'discard':
            card = select_card(self,hand, field)
            if card is None:
                return find_matching_card(self, hand, field)
            else:
                return card

        elif game_state == 'koikoi':
            if legal_actions[0] is None:
                return legal_actions[0]
            else:
                return decide_koikoi(self,your_yaku, your_score, op_score)
        else:
            return random.choice(legal_actions)


    def update_models(self):
        if len(self.card_data) > 1000:  # 一定量のデータが集まったら更新
            self.card_model = self.train_card_model()
            self.save_card_data()
            self.save_models()
            self.card_data = []  # データをリセット

        if len(self.koikoi_data) > 1000:
            self.koikoi_model = self.train_koikoi_model()
            self.save_koikoi_data()
            self.save_models()
            self.koikoi_data = []  # データをリセット

if __name__ == "__main__":
    my_agent = MyAgent()  # 参加者が実装したプレイヤーをインスタンス化

    mode = int(
        input(
            "Enter mode (1 for playing against AI, 2 for playing against another client): "
        )
    )
    num_games = int(input("Enter number of games to play: "))
    player_name = input("Enter your player name: ")

    sio_client = SocketIOClient(
        ip="localhost",
        port=5000,
        namespace="/koi-koi",
        agent=my_agent,
        room_id=123,
        player_name=player_name,
        mode=mode,
        num_games=num_games,
    )
    sio_client.run()
    # sio.client.enter_room()