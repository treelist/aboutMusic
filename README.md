# aboutMusic
[Personal study] Preprocessing for music analysis

음악 분석을 위한 전처리 과정을 만드는 것을 목표로 하는 개인 프로젝트 입니다.

최종 목표는 임의의 음악을 입력받았을 때 최종적으로 1. 박자 혹은 마디 단위의 코드 진행 배열, 2. 박자 단위의 세기 변화, 3. 음악의 속도를 출력하는 것입니다.

## 동기
음악을 많은 들어 본 사람은 직관적으로 듣고 있는 음악이 어떤 장르의 음악인지, 어떤 느낌을 주는 음악인지 알 수 있습니다. 그리고 그러한 '느낌'은 보편적으로 공유되는 것으로 보입니다.
[브금저장소](https://bgmstore.net/)는 익명의 다수가 음악의 느낌을 공유하는 하나의 예시입니다.

그렇다면 동일한 '느낌'을 주는 곡들을 머신 러닝 기법으로 학습시킨다면 머신은 새로운 곡에 대해 느낌으로 분류할 수 있을 것입니다. 
그리고 역으로 어떤 부분이 음악에서 그런 느낌을 만드는 부분인지 찾을 수 있을 것입니다. 더 나아가 필요한 느낌을 주는 곡을 만드는데 도움을 줄 수 도 있습니다.

음악을 학습시키기 위해 컴퓨터에게 음악을 입력해야 합니다. 컴퓨터에서의 음악은 기본적으로 Sampling rate 단위로 변위를 기록한 배열이라고 볼 수 있습니다. 
그 배열의 크기가 매우 크다는 것을 둘째로 치더라도, 곡마다 재생 시간도 다르고, 사용하는 악기도 다르고, 녹음환경도 다르기에 정규화 해줄 과정이 필요합니다.

차원 축소([Dimensionality Reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)), 
특징 선택([Feature Selection](https://en.wikipedia.org/wiki/Feature_selection)) 등을 이용하는 것도 한 방법일 수 있습니다.
그러나 음악적인 부분에서 접근을 시도해 보겠습니다.

## 범위 설정
음악을 구성하는 요소들에 대해서는 명확하게 정해진 것이 없는 듯 문헌마다 조금 차이를 보이기도 합니다. 
그러나  이 프로젝트에서는 느낌에 초점을 맞추어 코드진행, 박자의 세기 변화, 음악의 속도 이 세 가지에 초점을 맞추려고 합니다.

코드진행은 보편적으로 공유되는 음악의 느낌과 밀접한 관련이 있다고 생각합니다. 멜로디가 다르더라도 코드의 진행이 동일하면 비슷한 느낌을 받을 수 있습니다. 
그리고 코드진행은 상당히 정형화 되어 있습니다. ([네이버 책](http://book.naver.com/search/search.nhn?query=%EC%BD%94%EB%93%9C+%EC%A7%84%ED%96%89))
따라서 현대 대중 음악에 한정한다면 비슷한 느낌의 곡들이 공유하는 진행 부분이 많을 것으로 생각합니다.
그리고 [이 분석](http://www.hooktheory.com/blog/i-analyzed-the-chords-of-1300-popular-songs-for-patterns-this-is-what-i-found/)에 따르면 특정 코드(화음) 뒤에 나오는 코드의 비율은 통계적으로 일정하지 않습니다. 
이 사실은 나중에 코드를 정해기 애매한 경우가 발생할 때 [은닉 마르코프 모델](https://ko.wikipedia.org/wiki/%EC%9D%80%EB%8B%89_%EB%A7%88%EB%A5%B4%EC%BD%94%ED%94%84_%EB%AA%A8%EB%8D%B8)이나 [나이브 베이즈 분류](https://ko.wikipedia.org/wiki/%EB%82%98%EC%9D%B4%EB%B8%8C_%EB%B2%A0%EC%9D%B4%EC%A6%88_%EB%B6%84%EB%A5%98)를 구성 하여 통계적으로 접근할 근거가 되어줄 것입니다.

세기 변화도 주목할 필요가 있는 요소라고 생각합니다. 세기가 점점 커질 때는 감정이 고조되는 부분이고, 점점 작아질 때는 그 반대의 느낌을 갖습니다.

음악의 속도 느낌을 결정하는 중요한 요소입니다. 아무리 신나는 곡이라도 느리게 연주하면 신나는 느낌이 사라집니다. 

물론 이 외에도 다른 요소가 있을 것입니다. 생각해 볼 수 있는 요소는 음악의 [음색(timbre)](https://en.wikipedia.org/wiki/Timbre)과 [질감(texture)](https://en.wikipedia.org/wiki/Texture_(music))입니다. 
오케스트라가 연주한 웅장한 음악도 리코더로 연주하면 같은 느낌이 나기 힘듭니다. 생각해 볼 수 있는 주제이지만 이 프로젝트에서는 배제합니다.

따라서 최종 목표는 임의의 음악에 대해 그 음악의 코드진행, 세기변화, 속도를 구하는 것입니다. 

그리고 시작부터 많은 음악을 다루기는 힘들것이라 생각합니다. 그래서 사용하는 악기가 비슷하고 코드가 정형화되어 있는 20세기 이후 대중 음악(팝, 락, 발라드 등)을 위주로 합니다. 
그리고 마디가 뚜렷하지 않은 음악들(못갖춘마디로 시작, 도중 박자가 바뀜, 늘임표(페르마타) 존재)은 배제합니다.

## 계획
전체적인 진행 계획은 다음과 같습니다.
1. 박자, 마디 단위로 쪼개기 (진행중)
2. 정규화 및 노이즈필터링
3. 고속 푸리에 변환(Fast Fourier Transform)적용 및 이산화(Discretize) (old)
4. 이산화 되어 박자, 마디 구간과 코드 매칭
* 4-1. 각 구간과 코드 학습 (old)

## 참고
[관련 슬라이드](https://docs.google.com/presentation/d/1KDuoj-8nOaNf481Aq_ga0E4G8MMfLYw4z-y0FZ8nEDA/edit?usp=sharing)