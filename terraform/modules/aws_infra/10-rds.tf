# RDS subnet

resource "aws_db_subnet_group" "db_subnet_group" {
  count = var.rds_enabled ? 1 : 0
  name  = "${var.project}-db-subnet-group"
  subnet_ids = [
    #    aws_subnet.public_subnet_a.id,
    #    aws_subnet.public_subnet_b.id,
    #    aws_subnet.public_subnet_c.id,
    aws_subnet.private_subnet_a.id,
    aws_subnet.private_subnet_b.id,
    aws_subnet.private_subnet_c.id
  ]
}

resource "aws_db_subnet_group" "db_dataset_subnet_group" {
  count = var.rds_enabled ? 1 : 0
  name  = "${var.project}-db-dataset-subnet-group"
  subnet_ids = [
    aws_subnet.public_subnet_a.id,
    aws_subnet.public_subnet_b.id,
    aws_subnet.public_subnet_c.id
    #     aws_subnet.private_subnet_a.id,
    #     aws_subnet.private_subnet_b.id,
    #     aws_subnet.private_subnet_c.id
  ]
}

# RDS

resource "aws_db_parameter_group" "db_common_parameter_group" {
  name   = "db-common-parameter-group"
  family = "postgres16"

  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }
}

resource "aws_db_instance" "db_master" {
  count = var.rds_enabled ? 1 : 0

  identifier              = "${var.project}-db"
  engine                  = "postgres"
  engine_version          = "16.2"
  db_name                 = var.project
  instance_class          = "db.t4g.micro"
  storage_type            = "gp2"
  storage_encrypted       = true
  allocated_storage       = 10
  backup_retention_period = 7
  availability_zone       = "${var.aws_region}a"
  maintenance_window      = "Sun:04:00-Sun:06:00"
  backup_window           = "02:00-03:00"
  apply_immediately       = true
  network_type            = "DUAL"
  ca_cert_identifier      = "rds-ca-rsa2048-g1"
  db_subnet_group_name    = aws_db_subnet_group.db_subnet_group[count.index].name
  vpc_security_group_ids  = [aws_security_group.postgres_sg.id]
  username                = var.db_username
  password                = var.db_password
  skip_final_snapshot     = true

  tags = {
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_db_instance" "db_master_develop" {
  count = var.rds_enabled ? 1 : 0

  identifier              = "${var.project}-db-develop"
  engine                  = "postgres"
  engine_version          = "16.2"
  db_name                 = var.project
  instance_class          = "db.t4g.micro"
  storage_type            = "gp2"
  storage_encrypted       = true
  allocated_storage       = 10
  backup_retention_period = 7
  availability_zone       = "${var.aws_region}a"
  maintenance_window      = "Sun:04:00-Sun:06:00"
  backup_window           = "02:00-03:00"
  apply_immediately       = true
  network_type            = "DUAL"
  ca_cert_identifier      = "rds-ca-rsa2048-g1"
  db_subnet_group_name    = aws_db_subnet_group.db_subnet_group[count.index].name
  vpc_security_group_ids  = [aws_security_group.postgres_sg.id]
  username                = var.db_username
  password                = var.db_password
  skip_final_snapshot     = true

  tags = {
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_db_instance" "db_datasets" {
  count = var.rds_enabled ? 1 : 0

  identifier              = "${var.project}-db-datasets"
  engine                  = "postgres"
  engine_version          = "16.2"
  db_name                 = "datasets"
  instance_class          = "db.t4g.micro"
  parameter_group_name    = aws_db_parameter_group.db_common_parameter_group.name
  storage_type            = "gp2"
  storage_encrypted       = true
  allocated_storage       = 10
  backup_retention_period = 2
  publicly_accessible     = true
  availability_zone       = "${var.aws_region}a"
  maintenance_window      = "Sun:04:00-Sun:06:00"
  backup_window           = "02:00-03:00"
  apply_immediately       = true
  #   network_type            = "DUAL"
  ca_cert_identifier     = "rds-ca-rsa2048-g1"
  db_subnet_group_name   = aws_db_subnet_group.db_dataset_subnet_group[count.index].name
  vpc_security_group_ids = [aws_security_group.postgres_sg.id]
  username               = var.dataset_db_username
  password               = var.dataset_db_password
  skip_final_snapshot    = true

  tags = {
    Organization = var.organization
    Project      = var.project
  }
}

resource "null_resource" "db_setup" {
  depends_on = [aws_db_instance.db_datasets]

  provisioner "local-exec" {
    command = "psql -h ${aws_db_instance.db_datasets[0].address} -U ${var.dataset_db_username} -c 'CREATE DATABASE labelstudio;' postgres"
    environment = {
      PGPASSWORD = var.dataset_db_password
    }
  }
}