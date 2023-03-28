# Generated by Django 4.1.7 on 2023-03-28 06:18

from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Novel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=100, null=True)),
                ('cover_img', models.ImageField(null=True, upload_to='')),
                ('introduction', models.TextField(null=True)),
                ('status', models.IntegerField(choices=[(1, 'Finished'), (2, 'Pending'), (3, ' Wait_for_write')], default=3)),
                ('step', models.IntegerField(default=1, validators=[django.core.validators.MaxValueValidator(6), django.core.validators.MinValueValidator(1)])),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('genre', models.IntegerField(choices=[(1, 'romance'), (2, 'fantasy'), (3, 'mystery'), (4, 'sf'), (5, 'free')])),
                ('prompt', models.TextField()),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='NovelContent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField()),
                ('step', models.IntegerField(validators=[django.core.validators.MaxValueValidator(6), django.core.validators.MinValueValidator(1)])),
                ('query1', models.TextField()),
                ('query2', models.TextField()),
                ('query3', models.TextField()),
                ('novel', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='novels.novel')),
            ],
        ),
        migrations.CreateModel(
            name='NovelStats',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('hit_count', models.PositiveIntegerField(default=0)),
                ('comment_count', models.PositiveIntegerField(default=0)),
                ('like_count', models.PositiveIntegerField(default=0)),
                ('novel', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='novels.novel')),
            ],
        ),
        migrations.CreateModel(
            name='NovelLike',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('novel', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='novels.novel')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='NovelContentImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='')),
                ('caption', models.CharField(blank=True, max_length=255, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('novel_content', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='novels.novelcontent')),
            ],
        ),
        migrations.CreateModel(
            name='NovelComment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('novel', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='novels.novel')),
            ],
        ),
        migrations.AddConstraint(
            model_name='novellike',
            constraint=models.UniqueConstraint(fields=('novel', 'user'), name='unique_novel_user'),
        ),
        migrations.AddConstraint(
            model_name='novelcontent',
            constraint=models.UniqueConstraint(fields=('novel', 'step'), name='unique_novel_step'),
        ),
    ]
