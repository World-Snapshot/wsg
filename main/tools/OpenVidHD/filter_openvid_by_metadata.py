import pandas as pd
import re

def extract_youtube_id_from_filename(video_filename):
    """从OpenVidHD.csv的video列提取YouTube ID"""
    # 格式如: hazuxwlzQNg_42_18to125.mp4
    name = video_filename.replace('.mp4', '')
    parts = name.split('_')
    
    if len(parts) >= 1:
        youtube_id = parts[0]
        return youtube_id
    
    return None

def filter_openvid_by_metadata():
    """根据metadata_full.csv筛选OpenVidHD.csv"""
    
    # 读取metadata_full.csv，提取所有videoID
    print("读取metadata_full.csv...")
    metadata_df = pd.read_csv('metadata_full.csv')
    metadata_video_ids = set(metadata_df['videoID'].tolist())
    print(f"metadata_full.csv包含 {len(metadata_df)} 条记录")
    print(f"唯一videoID数量: {len(metadata_video_ids)}")
    
    # 读取OpenVidHD.csv
    print("\\n读取OpenVidHD.csv...")
    openvid_df = pd.read_csv('OpenVidHD.csv')
    print(f"OpenVidHD.csv包含 {len(openvid_df)} 条记录")
    
    # 筛选匹配的OpenVidHD条目
    print("\\n筛选匹配的OpenVidHD条目...")
    filtered_rows = []
    
    for idx, row in openvid_df.iterrows():
        if idx % 50000 == 0:
            print(f"  处理进度: {idx}/{len(openvid_df)}")
        
        video_filename = row['video']
        youtube_id = extract_youtube_id_from_filename(video_filename)
        
        # 如果YouTube ID在metadata中存在，就保留这一行
        if youtube_id in metadata_video_ids:
            filtered_rows.append(row)
    
    # 创建筛选后的DataFrame
    if filtered_rows:
        filtered_df = pd.DataFrame(filtered_rows)
        
        # 保存结果
        output_filename = '筛选后的OpenVidHD.csv'
        filtered_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        print(f"\\n筛选完成！")
        print(f"原始OpenVidHD.csv: {len(openvid_df)} 条记录")
        print(f"筛选后的OpenVidHD.csv: {len(filtered_df)} 条记录")
        print(f"保留比例: {len(filtered_df)/len(openvid_df)*100:.2f}%")
        print(f"结果已保存到: {output_filename}")
        
        # 统计信息
        print(f"\\n统计信息:")
        unique_video_ids = set()
        for _, row in filtered_df.iterrows():
            youtube_id = extract_youtube_id_from_filename(row['video'])
            if youtube_id:
                unique_video_ids.add(youtube_id)
        
        print(f"筛选后唯一YouTube视频ID数量: {len(unique_video_ids)}")
        print(f"metadata中的唯一videoID数量: {len(metadata_video_ids)}")
        print(f"匹配的视频ID数量: {len(unique_video_ids)}")
        
        # 相机运动类型分布
        if 'camera motion' in filtered_df.columns:
            print(f"\\n相机运动类型分布:")
            camera_motion_counts = filtered_df['camera motion'].value_counts()
            for motion_type, count in camera_motion_counts.head(10).items():
                print(f"  {motion_type}: {count} 条记录")
        
        # 显示一些示例
        print(f"\\n筛选结果示例:")
        for i, (_, row) in enumerate(filtered_df.head(5).iterrows()):
            youtube_id = extract_youtube_id_from_filename(row['video'])
            print(f"{i+1}. 视频ID: {youtube_id}")
            print(f"   文件名: {row['video']}")
            print(f"   时长: {row['seconds']:.1f}秒")
            print(f"   相机运动: {row['camera motion']}")
            print(f"   美学评分: {row['aesthetic score']:.2f}")
            print()
    else:
        print("\\n没有找到匹配的记录！")

if __name__ == "__main__":
    filter_openvid_by_metadata()