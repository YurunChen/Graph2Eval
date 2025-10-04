USE wordpress;

-- 删除重复的帖子，保留ID最小的
DELETE p1 FROM wp_posts p1
INNER JOIN wp_posts p2 
WHERE p1.ID > p2.ID 
AND p1.post_title = p2.post_title 
AND p1.post_type = p2.post_type 
AND p1.post_status = p2.post_status;

-- 删除重复的页面，保留ID最小的
DELETE p1 FROM wp_posts p1
INNER JOIN wp_posts p2 
WHERE p1.ID > p2.ID 
AND p1.post_title = p2.post_title 
AND p1.post_type = 'page' 
AND p2.post_type = 'page'
AND p1.post_status = p2.post_status;

-- 删除重复的评论（基于评论内容和文章ID）
DELETE c1 FROM wp_comments c1
INNER JOIN wp_comments c2 
WHERE c1.comment_ID > c2.comment_ID 
AND c1.comment_content = c2.comment_content 
AND c1.comment_post_ID = c2.comment_post_ID;

-- 删除重复的分类和标签
DELETE t1 FROM wp_terms t1
INNER JOIN wp_terms t2 
WHERE t1.term_id > t2.term_id 
AND t1.name = t2.name 
AND t1.slug = t2.slug;

-- 删除重复的分类关系
DELETE tr1 FROM wp_term_relationships tr1
INNER JOIN wp_term_relationships tr2 
WHERE tr1.object_id = tr2.object_id 
AND tr1.term_taxonomy_id = tr2.term_taxonomy_id 
AND tr1.object_id > tr2.object_id;

-- 重新计算评论数量
UPDATE wp_posts p 
SET comment_count = (
    SELECT COUNT(*) 
    FROM wp_comments c 
    WHERE c.comment_post_ID = p.ID 
    AND c.comment_approved = '1'
);

-- 重新计算分类和标签的数量
UPDATE wp_term_taxonomy tt 
SET count = (
    SELECT COUNT(*) 
    FROM wp_term_relationships tr 
    WHERE tr.term_taxonomy_id = tt.term_taxonomy_id
);

-- 显示清理结果
SELECT 'Duplicate cleanup completed!' as status;
SELECT COUNT(*) as total_posts FROM wp_posts WHERE post_type = 'post' AND post_status = 'publish';
SELECT COUNT(*) as total_pages FROM wp_posts WHERE post_type = 'page' AND post_status = 'publish';
SELECT COUNT(*) as total_comments FROM wp_comments WHERE comment_approved = '1';
SELECT COUNT(*) as total_categories FROM wp_term_taxonomy WHERE taxonomy = 'category';
SELECT COUNT(*) as total_tags FROM wp_term_taxonomy WHERE taxonomy = 'post_tag';


