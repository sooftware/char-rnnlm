
class KoNPron:
    """ Python package for Korean Number Pronunciation """
    def __init__(self):
        self.base_digit = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        self.super_digit = ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')
        self.small_scale = ('', '십', '백', '천')
        self.large_scale = ('', '만 ', '억 ', '조 ', '경 ', '해 ')
        self.literal = ('영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구')
        self.spoken_unit = ('', '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉')
        self.spoken_tens = ('', '열', '스물', '서른', '마흔', '쉰', '예순', '일흔', '여든', '아흔')

    def detect(self, sentence):
        detection_data = list()
        tmp = str()

        total_len = len(sentence)
        point_count = 0
        digit_type = 'vanilla'
        detected = False

        for idx, ch in enumerate(sentence):
            if ch in self.base_digit:
                if not detected:
                    detected = True
                tmp += ch

            else:
                if ch == ',':
                    if idx + 1 < total_len and idx > 0:
                        if sentence[idx - 1] in self.base_digit and sentence[idx + 1] in self.base_digit:
                            tmp += ch

                elif ch == '.':
                    if idx + 1 < total_len and idx > 0:
                        if sentence[idx - 1] in self.base_digit and sentenc[idx + 1] in self.base_digit:
                            point_count += 1

                            if point_count == 1:
                                digit_type = 'fraction'

                            if point_count > 1:
                                digit_type = 'version'

                            tmp += ch

                elif ch == '^':
                    if idx + 1 < total_len:
                        if sentence[idx + 1] in self.base_digit:
                            digit_type += '/square'
                            tmp += ch

                elif ch in self.super_digit:
                    if idx > 0:
                        if sentence[idx - 1] in self.base_digit:
                            digit_type += '/square'
                            tmp += ch

                else:
                    if detected:
                        detected = False
                        detection_data.append((tmp, digit_type))
                        tmp = str()
                        digit_type = 'vanilla'

        detection_data.append((tmp, digit_type))

        return detection_data

    def preprocess(self, detection_data, mode='spoken'):
        target, digit_type = detection_data
        length = len(target)

        korean = str()
        main_type, optional_type = digit_type.split('/')

        if main_type == 'version':

        if main_type == 'vanila':
            target = target.replace(',', '')

        if main_type == 'fraction':
            point_idx = target.index('.')
            fraction = target[point_idx + 1:]
            integer = target[:point_idx]
            length = len(integer)