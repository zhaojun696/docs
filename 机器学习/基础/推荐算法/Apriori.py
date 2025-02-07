#%%
from itertools import combinations

class Apriori:
    def __init__(self, min_support, min_confidence,load_data=None):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []
        if(load_data):self.load_data=load_data

    def load_data(self):
        # 示例数据集
        data = [
            ['牛奶', '面包', '黄油'],
            ['牛奶', '面包', '尿布', '啤酒'],
            ['牛奶', '尿布', '啤酒', '可乐'],
            ['面包', '黄油', '尿布', '啤酒'],
            ['面包', '尿布', '啤酒']
        ]
        return data

    def create_candidates(self, data, k):
        candidates = set()
        for transaction in data:
            for item in transaction:
                candidates.add(frozenset([item]))
        return [frozenset(item) for item in candidates]

    def filter_candidates(self, data, candidates):
        item_count = {}
        for transaction in data:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    if candidate not in item_count:
                        item_count[candidate] = 1
                    else:
                        item_count[candidate] += 1
        
        num_transactions = len(data)
        frequent_items = {item: count / num_transactions for item, count in item_count.items() if count / num_transactions >= self.min_support}
        return frequent_items

    def generate_frequent_itemsets(self, data, max_length=3):
        k = 1
        candidates = self.create_candidates(data, k)
        
        while candidates and k <= max_length:
            frequent_items = self.filter_candidates(data, candidates)
            self.frequent_itemsets.update(frequent_items)
            
            k += 1
            candidates = [frozenset(comb) for comb in combinations(set().union(*frequent_items.keys()), k)]

    def generate_rules(self):
        for itemset in self.frequent_itemsets:
            if len(itemset) > 1:
                for i in range(1, len(itemset)):
                    for subset in combinations(itemset, i):
                        antecedent = frozenset(subset)
                        consequent = itemset - antecedent
                        confidence = self.frequent_itemsets[itemset] / self.frequent_itemsets[antecedent]
                        if confidence >= self.min_confidence:
                            self.rules.append((antecedent, consequent, confidence))

    def run(self):
        data = self.load_data()
        self.generate_frequent_itemsets(data)
        self.generate_rules()

    def print_results(self):
        print("Frequent Itemsets:")
        for itemset, support in self.frequent_itemsets.items():
            print(f"{set(itemset)}: {support:.2f}")
        
        print("\nAssociation Rules:")
        for antecedent, consequent, confidence in self.rules:
            print(f"{set(antecedent)} => {set(consequent)}: {confidence:.2f}")

apriori = Apriori(min_support=0.6, min_confidence=0.7)
apriori.run()
apriori.print_results()


#%%
# Frequent Itemsets（频繁项集）
# 频繁项集是指在数据集中出现频率高于某个最小支持度阈值的项集。在这个例子中，最小支持度阈值是0.6。
# 单项频繁项集：
# {'牛奶'}: 0.60
# {'面包'}: 0.80
# {'啤酒'}: 0.80
# {'尿布'}: 0.80
# 这些项在数据集中出现的频率都超过了最小支持度阈值0.6。
# 二项频繁项集：
# {'尿布', '面包'}: 0.60
# {'尿布', '啤酒'}: 0.80
# {'面包', '啤酒'}: 0.60
# 这些二项集在数据集中出现的频率也都超过了最小支持度阈值0.6。
# 三项频繁项集：
# {'尿布', '面包', '啤酒'}: 0.60
# 这个三项集在数据集中出现的频率超过了最小支持度阈值0.6。

# Association Rules（关联规则）
# 关联规则是指从频繁项集中生成的规则，这些规则的置信度高于某个最小置信度阈值。在这个例子中，最小置信度阈值是0.7。
# 单项到单项的关联规则：
# {'尿布'} => {'面包'}: 0.75
# {'面包'} => {'尿布'}: 0.75
# {'尿布'} => {'啤酒'}: 1.00
# {'啤酒'} => {'尿布'}: 1.00
# {'面包'} => {'啤酒'}: 0.75
# {'啤酒'} => {'面包'}: 0.75
# 这些规则表示，如果一个顾客购买了规则左侧的项，那么他们也有很大可能性购买规则右侧的项。例如，{'尿布'} => {'啤酒'} 表示购买尿布的顾客中有100%的可能性也会购买啤酒。
# 单项到二项的关联规则：
# {'尿布'} => {'面包', '啤酒'}: 0.75
# {'面包'} => {'尿布', '啤酒'}: 0.75
# {'啤酒'} => {'尿布', '面包'}: 0.75
# 这些规则表示，如果一个顾客购买了规则左侧的项，那么他们也有很大可能性购买规则右侧的二项集。例如，{'尿布'} => {'面包', '啤酒'} 表示购买尿布的顾客中有75%的可能性也会购买面包和啤酒。
# 二项到单项的关联规则：
# {'尿布', '面包'} => {'啤酒'}: 1.00
# {'尿布', '啤酒'} => {'面包'}: 0.75
# {'面包', '啤酒'} => {'尿布'}: 1.00
# 这些规则表示，如果一个顾客购买了规则左侧的二项集，那么他们也有很大可能性购买规则右侧的项。例如，{'尿布', '面包'} => {'啤酒'} 表示购买尿布和面包的顾客中有100%的可能性也会购买啤酒。
# 通过这些频繁项集和关联规则，我们可以了解哪些商品经常一起被购买，从而帮助商家进行商品推荐或促销策略的制定。

