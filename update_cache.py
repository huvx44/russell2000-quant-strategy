"""
캐시 업데이트 유틸리티
====================
주식 데이터 캐시를 강제로 새로고침합니다.

사용법:
    python update_cache.py

이 스크립트는 generate_portfolio.py를 --refresh 모드로 실행하여
최신 데이터를 다운로드하고 캐시를 업데이트합니다.
"""

import subprocess
import sys

if __name__ == "__main__":
    print("=" * 70)
    print("     캐시 업데이트 시작")
    print("=" * 70)
    print("\n최신 데이터를 다운로드하고 캐시를 갱신합니다...")
    print("이 작업은 5-10분 정도 소요됩니다.\n")

    # generate_portfolio.py를 --refresh 모드로 실행
    result = subprocess.run([sys.executable, 'generate_portfolio.py', '--refresh'])

    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("     캐시 업데이트 완료!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("     오류 발생")
        print("=" * 70)
        sys.exit(1)
