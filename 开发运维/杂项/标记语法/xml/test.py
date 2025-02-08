xmlStr='''
<root>
    <describe>
    142个证券账户在交易涉案股票期间存在IP、MAC关联,
    并由陈霄自认、部分证券账户介绍人指认、部分证券账户与陈霄有签署协议。
    资金来源于账户配资资金及自有资金,交易由陈霄本人决策、由其负责的团队具体负责操作下单。
    </describe>
    <info>
        <count chart='line'>
            142
        </count>
        <startDate>
            20190315
        </startDate>
        <endDate>
            20200420
        </endDate>
    </info>
    <execList>
        
    </execList>
</root>
'''

import xml.etree.ElementTree as ET


# 解析XML字符串
root = ET.fromstring(xmlStr)
# root = ET.parse('file.xml').getroot()

# # 查看标签
# print(root.tag)
# # 查看属性
# print(root.attrib)
# # 查看文本,去除空格
# print(root.text.strip())


# # 根据标签寻找子元素,find总是找到第1个碰到的元素
# print(root.find('country'))
# # findall是找到所有的的元素
# print(root.findall('country'))
# # 不需要列表，希望是一个可迭代对象,得到一个生成器对象
# print(root.iterfind('country'))



# *能匹配所有的child,只想找root的所有孙子节点
# print(root.findall('country/*'))

# 查找任意层次下的子元素，.点为当前节点，..为父节点
# print(root.findall('.//rank'))
# print(root.findall('.//rank/..'))

# @描述包含某一属性，[@attrib]
# print(root.findall('country[@name]'))

# 指定属性为特定值，[@attrib='value']
# print(root.findall('country[@name="Singapore"]'))

# 指定一个元素必须包含一个指定的子元素，[tag]
# print(root.findall('country[rank]'))

# 指定元素的子元素文本必须等于特定的值，[tag='text']
# print(root.findall('country[rank="5"]'))

# 找多个元素路径指定相对位置，[position]
# print(root.findall('country[1]'))
# print(root.findall('country[2]'))

# last()为倒着找
# print(root.findall('country[last()]'))
# 找倒数第二个
# print(root.findall('country[last()-1]'))


# # 获取根元素的子元素
# for child in root.find('info'):
#     print(child.tag, child.text,child)

root.find('info/count').text.strip()
root.find('info/count').get('chart')
