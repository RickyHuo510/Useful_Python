import requests
from bs4 import BeautifulSoup
import time
import datetime
import os
import sys

# ================= 配置区域 =================
# 【重要】请将你在浏览器F12中获取的Cookie粘贴在下面引号内
USER_COOKIE = 'Admin-Token=eyJhbGciOiJIUzUxMiJ9.eyJsb2dpbl91c2VyX2tleSI6IjVlZGI0MTY5LTNiNDktNGRiMy1hYWU5LTNhODhlMDIyNmIzNSJ9.IrovKObBtt8eZEE_f7ygxrhw96GDvTKCgaLjdFq-LJrjuDh8gT-aU6pYf4YGa55v5YC_Dq6dTv2HOxF440tJKA; sessionid=4knp6urqhuoytnqhofh3vjw62pjn8xmf; csrftoken=NsCUgBHGkVA7jfinLo53VJ1iCtE03avV'

# 目标课程名称片段
TARGET_COURSE = "计算机网络与通信技术"

# 刷新间隔 (秒) - 警告：设置为1秒极易被封IP，建议至少3秒
REFRESH_INTERVAL = 2 

# 目标URL (根据你提供的HTML文件中的URL)
URL = "https://aa.bjtu.edu.cn/course_selection/courseselecttask/selects/"

# 请求头，伪装成浏览器
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Cookie': USER_COOKIE,
    'Referer': 'https://aa.bjtu.edu.cn/course_selection/courseselecttask/selects/'
}
# ===========================================

def beep():
    """触发系统提示音"""
    # Windows
    if sys.platform == 'win32':
        import winsound
        winsound.Beep(1000, 1000) # 频率1000Hz，持续1秒
    else:
        # Mac/Linux
        print('\a') 

def monitor():
    print(f"[*] 开始监控课程: {TARGET_COURSE}")
    print(f"[*] 监控链接: {URL}")
    print(f"[*] 按 Ctrl+C 停止")
    
    count = 0
    
    while True:
        try:
            count += 1
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            
            # 1. 发起请求
            response = requests.get(URL, headers=HEADERS, timeout=5)
            
            # 2. 检查Cookie是否失效 (通常失效会重定向到登录页)
            if "login" in response.url or response.status_code != 200:
                print(f"\n[!] {current_time} Cookie可能已失效或网络错误，状态码: {response.status_code}")
                print("[!] 请重新在浏览器获取Cookie并更新脚本。")
                break
                
            # 3. 解析网页
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 定位到"本学期课程"的表格区域 (id="current")
            current_tab = soup.find('div', id='current')
            if not current_tab:
                # 尝试直接找table，防止HTML结构微调
                tables = soup.find_all('table', class_='table')
                if tables:
                    main_table = tables[-1] # 假设最后一个表格
                else:
                    print(f"[-] {current_time} 未找到课程列表表格，可能页面加载失败。")
                    continue
            else:
                main_table = current_tab.find('table')

            if not main_table:
                print(f"[-] {current_time} 表格未找到")
                continue

            # 4. 遍历所有行寻找目标课程
            rows = main_table.find_all('tr')
            found_target = False
            
            for row in rows:
                cols = row.find_all('td')
                if not cols: 
                    continue # 跳过表头
                
                # 根据HTML结构：
                # 第2列 (index 1) 是课程名称信息
                # 第3列 (index 2) 是课余量
                
                course_info = cols[1].get_text(strip=True)
                spots_text = cols[2].get_text(strip=True)
                
                if TARGET_COURSE in course_info:
                    found_target = True
                    
                    try:
                        spots = int(spots_text)
                    except ValueError:
                        spots = 0
                        
                    status_msg = f"[{current_time}] 第{count}次扫描 | {course_info[:20]}... | 余量: {spots}"
                    
                    if spots > 0:
                        # === 发现余量，强烈提醒 ===
                        print("\n" + "="*50)
                        print(f"!!! 恭喜！发现余量 !!!")
                        print(f"课程: {course_info}")
                        print(f"余量: {spots}")
                        print("="*50 + "\n")
                        
                        # 疯狂响铃提醒
                        for _ in range(3):
                            beep()
                            time.sleep(0.5)
                    else:
                        # 没有余量，覆盖打印一行
                        print(status_msg, end='\r')

            if not found_target:
                print(f"\n[-] {current_time} 列表中未找到名称包含 '{TARGET_COURSE}' 的课程，请检查关键词是否正确。")
            
            # 等待下一次刷新
            time.sleep(REFRESH_INTERVAL)
            
        except requests.exceptions.RequestException as e:
            print(f"\n[!] 网络请求出错: {e}")
            time.sleep(REFRESH_INTERVAL)
        except Exception as e:
            print(f"\n[!] 发生未知错误: {e}")
            time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    if "sessionid" not in USER_COOKIE:
        print("警告：Cookie看起来未填写或格式不正确，脚本可能无法运行。")
    monitor()