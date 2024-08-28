from cgi_drl.problem.supervised_learning_trainer import SupervisedLearningTrainer
import tensorflow as tf
import json

# game_counts = {'Treant Druid': 39786, 'Plague Death Knight': 33720, 'Mining Rogue': 27988, 'Unholy Death Knight': 25814, 'Blood Death Knight': 21754, 'Dragon Druid': 20242, 'Control Warrior': 16990, 'Mining Warrior': 14708, 'Mining Mage': 14338, 'Aggro Paladin': 13470, 'Highlander Druid': 11654, 'Automaton Priest': 11616, 'Rainbow Mage': 10840, 'Control Priest': 10418, 'Earthen Paladin': 9762, 'Sludge Warlock': 9550, 'Thaddius Warlock': 9058, 'Big Demon Hunter': 8378, 'Mech Rogue': 6264, 'Highlander Shaman': 6002, 'Wishing Rogue': 5338, 'Arcane Hunter': 5314, 'Undead Priest': 5214, 'Secret Mage': 4822, 'Highlander Hunter': 4658, 'Curse Warlock': 4560, "Rock 'n' Roll Warrior": 4470, 'Rainbow Death Knight': 3518, 'Mining Warlock': 3500, 'Highlander Paladin': 3466, 'Highlander Blood Death Knight': 3222, 'Aggro Demon Hunter': 3112, 'Highlander Priest': 3094, 'Nature Druid': 3046, 'Elemental Mage': 2804, 'Hound Hunter': 2658, 'Cleave Hunter': 2654, 'Highlander Warlock': 2362, 'Totem Shaman': 2122, 'Elemental Shaman': 1536, 'Nature Shaman': 1448, 'Taunt Warrior': 1110, 'Naga Priest': 1058, 'Enrage Warrior': 992, 'Silver Hand Paladin': 976, 'Showdown Paladin': 942, 'Ogre Rogue': 808, 'Highlander Mage': 780, 'Relic Demon Hunter': 636, 'Thaddius Druid': 630, 'Highlander Rogue': 604, 'Highlander Demon Hunter': 550, 'Miracle Rogue': 440, 'Mech Mage': 358, 'Highlander Warrior': 252, 'Spooky Mage': 216, 'Big Rogue': 140, 'Frost Death Knight': 124, 'Control Warlock': 60, 'Pure Paladin': 42, 'Secret Rogue': 42, 'Secret Hunter': 30, 'Weapon Rogue': 30, 'Mech Paladin': 30, 'Evolve Shaman': 24, 'Ogre Priest': 24, 'Drum Druid': 24, 'Lightshow Mage': 10, 'Moonbeam Druid': 10, 'Breakfast Hunter': 8, 'Murloc Warlock': 8, 'Control Mage': 8, 'Big Shaman': 8, 'Imp Warlock': 8, 'Burn Mage': 6, 'Combo Rogue': 6, 'Dancing Paladin': 6, 'Naga Demon Hunter': 6, 'Face Hunter': 4, 'Ramp Druid': 4, 'Mill Druid': 2, 'Miracle Priest': 2, 'Overload Shaman': 2, 'Big Druid': 2, 'Menagerie Warrior': 2, 'Frost Aggro Death Knight': 2, 'Casino Mage': 2, 'Deathrattle Druid': 2, 'Yogg Shaman': 2, 'Outcast Demon Hunter': 2, 'Spell Demon Hunter': 2}

class EnumerateRatingAndCounterTableSolver(SupervisedLearningTrainer):
    def initialize(self):
        self.summary_writer = tf.summary.FileWriter(self.log_path)
        self.log_file = open(self.log_path + "/log.txt", 'a', 1)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()
        self.rating_table_model.set_session(self.sess)
        self.counter_table_model.set_session(self.sess)

        self.batch_size = self.solver_config["batch_size"]

        self.sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ])

        self.rating_table_model_path = self.solver_config["rating_table_model_path"]
        self.counter_table_model_path = self.solver_config["counter_table_model_path"]
        
        self.rating_table_model.load(self.rating_table_model_path)
        self.counter_table_model.load(self.counter_table_model_path)

        self.rating_table_file_path = self.log_path + "/rating_table.json"
        self.counter_table_file_path = self.log_path + "/counter_table.json"

    def terminate(self):
        pass

    def on_epoch(self):
        rating_table = dict() # category name rating
        counter_table = dict()

        for records in self.result_buffer.sample_all_combos(self.batch_size):
            sample_count = len(records["player1_combos"])
            comp1_rating, comp2_rating, bt_winvalue_prediction = self.rating_table_model.get_rating_and_predictions({"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]})
            winvalue_prediction, player1_embedding_k, player2_embedding_k = self.counter_table_model.get_predictions({"player1_combo":records["player1_combos"], "player2_combo":records["player2_combos"]}, bt_winvalue_prediction)
            for i in range(sample_count):
                k1, k2 = int(player1_embedding_k[i]), int(player2_embedding_k[i])
                # match_result = records["match_results"][i]
                # if game_counts[records["player1_raw_combos"][i]] < 100:
                #     continue
                # if game_counts[records["player2_raw_combos"][i]] < 100:
                #     continue

                # if match_result > 0.501: # win
                #     if bt_winvalue_prediction[i] <= 0.501 and winvalue_prediction[i] > 0.501:
                #         pass
                #     else:
                #         continue
                # elif match_result < 0.499: # lose
                #     if bt_winvalue_prediction[i] >= 0.499 and winvalue_prediction[i] < 0.499:
                #         pass
                #     else:
                #         continue
                # else: # tie
                #     if not (bt_winvalue_prediction[i] >= 0.499 and bt_winvalue_prediction[i] <= 0.501):
                #         if winvalue_prediction[i] >= 0.499 and winvalue_prediction[i] <= 0.501:
                #             pass
                #         else:
                #             continue
                #     else:
                #         continue

                if k1 not in rating_table:
                    rating_table[k1] = dict()
                if k2 not in rating_table:
                    rating_table[k2] = dict()
                #if float(comp1_rating[i]) < 100:
                rating_table[k1][records["player1_raw_combos"][i]] = float(comp1_rating[i])
                #if float(comp2_rating[i]) < 100:
                rating_table[k2][records["player2_raw_combos"][i]] = float(comp2_rating[i])
                if k1 not in counter_table:
                    counter_table[k1] = dict()
                    counter_table[k1][k1] = 0
                if k2 not in counter_table:
                    counter_table[k2] = dict()
                    counter_table[k2][k2] = 0
                counter_table[k1][k2] = float(winvalue_prediction[i] - bt_winvalue_prediction[i])
                counter_table[k2][k1] = -float(winvalue_prediction[i] - bt_winvalue_prediction[i])
        for category in rating_table:
             rating_table[category] = dict(sorted(rating_table[category].items(), key=lambda x:x[1], reverse=True))
             rating_sum = 0
             for c in rating_table[category]:
                 rating_sum += rating_table[category][c]
             print(f"Category{category} has {len(rating_table[category])} compositions, average rating: {rating_sum/len(rating_table[category])}")
        
        with open(self.rating_table_file_path, 'w', encoding='utf-8') as file:
            json.dump(rating_table, file)
        with open(self.counter_table_file_path, 'w', encoding='utf-8') as file:
            json.dump(counter_table, file)

        ## Top-B
        top_b_balance = 0
        top_comps = []
        for category in rating_table:
            c, c_rating = sorted(rating_table[category].items(), key=lambda x:x[1], reverse=True)[0]
            print((c, c_rating, category))
            top_comps.append((c, c_rating, category))

        for c, c_rating, c_category in top_comps:
            dominated = False
            for cp, cp_rating, cp_category in top_comps:
                if c == cp:
                    continue
                domi = True
                for cpp, cpp_rating, cpp_category in top_comps:
                    if cp_rating / (cp_rating + cpp_rating) + counter_table[cp_category][cpp_category] <= c_rating / (c_rating + cpp_rating) + counter_table[c_category][cpp_category]:
                        domi = False
                        break
                if domi:
                    dominated = True
                    break
            if not dominated:
                top_b_balance += 1
            else:
                print(c, c_rating, c_category, " is dominated")
        print("Top-B Balance: ", top_b_balance)

        # Top-D
        max_rating = -999
        max_comp = None

        for category in rating_table:
            for comp in rating_table[category]:
                if rating_table[category][comp] > max_rating:
                    max_rating = rating_table[category][comp]
                    max_comp = (category, max_comp)
        print("Top Comp: ", max_comp, " ", max_rating)
        
        top_d_diversity = 0
        g = 0.01
        for category in rating_table:
            for comp in rating_table[category]:
                if (rating_table[category][comp] / (rating_table[category][comp] + max_rating) + g) >= 0.5:
                    top_d_diversity += 1
        print("Top-D Diversity G=0.01: ", top_d_diversity)

        top_d_diversity = 0
        g = 0.02
        for category in rating_table:
            for comp in rating_table[category]:
                if (rating_table[category][comp] / (rating_table[category][comp] + max_rating) + g) >= 0.5:
                    top_d_diversity += 1
        print("Top-D Diversity G=0.02: ", top_d_diversity)

        top_d_diversity = 0
        g = 0.04
        for category in rating_table:
            for comp in rating_table[category]:
                if (rating_table[category][comp] / (rating_table[category][comp] + max_rating) + g) >= 0.5:
                    top_d_diversity += 1
        print("Top-D Diversity G=0.04: ", top_d_diversity)

        top_d_diversity = 0
        g = 0.08
        for category in rating_table:
            for comp in rating_table[category]:
                if (rating_table[category][comp] / (rating_table[category][comp] + max_rating) + g) >= 0.5:
                    top_d_diversity += 1
        print("Top-D Diversity G=0.08: ", top_d_diversity)
